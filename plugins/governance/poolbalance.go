package governance

import (
	"math/rand/v2"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// poolBalancer decides which provider serves a model that several providers can serve, and
// remembers the answer per session.
//
// The point is that a vLLM prefix cache lives inside one instance: a conversation that moves
// between instances re-prefills its whole context on the new one. So a session should stay put.
// But pinning by a hash of the session id (what a routing rule can express) leaves the split to
// chance, and two sessions landing on the same provider leaves the other idle. This balancer
// assigns a session to the provider with the fewest live sessions, keeps it there, and lets the
// assignment expire when the session goes quiet so later sessions re-spread.
//
// It deliberately counts sessions rather than requests: no per-request state to increment and
// decrement, so nothing can leak and no hook pairing has to be exactly right.
type poolBalancer struct {
	mu       sync.Mutex
	sessions map[string]*poolSession

	// ttl is how long a session keeps its assignment after its last request.
	ttl time.Duration
	// rebalanceGap is how many more live sessions a provider may carry than the least loaded one
	// before a session is moved off it. 0 disables moving a session that is already assigned.
	rebalanceGap int
	now          func() time.Time
}

type poolSession struct {
	provider string
	lastSeen time.Time
}

func newPoolBalancer() *poolBalancer {
	return &poolBalancer{
		sessions:     make(map[string]*poolSession),
		ttl:          durationFromEnv("BIFROST_POOL_SESSION_TTL", 30*time.Minute),
		rebalanceGap: intFromEnv("BIFROST_POOL_REBALANCE_GAP", 2),
		now:          time.Now,
	}
}

// pick returns the candidate that should serve this request. An empty sessionKey means the
// request carries no session identity, and a spread across the candidates by weight is the
// best that can be done for it. The weights are used only to break ties.
func (b *poolBalancer) pick(sessionKey string, candidates []schemas.ProviderCandidate) (schemas.ProviderCandidate, string) {
	if len(candidates) == 0 {
		return schemas.ProviderCandidate{}, "no candidates"
	}
	if len(candidates) == 1 {
		return candidates[0], "only one candidate"
	}
	if sessionKey == "" {
		return weightedPick(candidates), "no session id: weighted pick"
	}
	if b == nil {
		return weightedPick(candidates), "no balancer: weighted pick"
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	now := b.now()
	live := b.liveCountsLocked(now)

	if session, ok := b.sessions[sessionKey]; ok {
		for _, candidate := range candidates {
			if candidate.Provider != session.provider {
				continue
			}
			// A session stays where it is unless its provider has run away with the load. Moving
			// it costs one full re-prefill, so the gap has to be worth that.
			least := b.leastLoadedLocked(candidates, live)
			if b.rebalanceGap > 0 && least != "" && live[session.provider]-live[least] >= b.rebalanceGap {
				chosen := pickByLoad(candidates, live)
				session.provider = chosen.Provider
				session.lastSeen = now
				return chosen, "rebalanced off an overloaded provider"
			}
			session.lastSeen = now
			return candidate, "sticky to the session's provider"
		}
		// The session's provider is not eligible for this model any more: fall through and assign
		// again, which also repairs the stale assignment.
	}

	chosen := pickByLoad(candidates, live)
	b.sessions[sessionKey] = &poolSession{provider: chosen.Provider, lastSeen: now}
	return chosen, "assigned to the least loaded provider"
}

// liveCountsLocked counts sessions per provider and drops the ones that have gone quiet, so the
// map cannot grow without bound and an idle provider becomes available again.
func (b *poolBalancer) liveCountsLocked(now time.Time) map[string]int {
	counts := make(map[string]int, len(b.sessions))
	for key, session := range b.sessions {
		if now.Sub(session.lastSeen) > b.ttl {
			delete(b.sessions, key)
			continue
		}
		counts[session.provider]++
	}
	return counts
}

// leastLoadedLocked names the provider with the fewest live sessions, or "" when there are none.
func (b *poolBalancer) leastLoadedLocked(candidates []schemas.ProviderCandidate, live map[string]int) string {
	least := ""
	leastCount := 0
	for _, candidate := range candidates {
		count := live[candidate.Provider]
		if least == "" || count < leastCount {
			least, leastCount = candidate.Provider, count
		}
	}
	return least
}

// pickByLoad chooses among the candidates with the fewest live sessions, breaking that tie by
// weight so several equally idle providers still see traffic in proportion to their weight.
func pickByLoad(candidates []schemas.ProviderCandidate, live map[string]int) schemas.ProviderCandidate {
	fewest := -1
	for _, candidate := range candidates {
		if count := live[candidate.Provider]; fewest < 0 || count < fewest {
			fewest = count
		}
	}
	tied := make([]schemas.ProviderCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if live[candidate.Provider] == fewest {
			tied = append(tied, candidate)
		}
	}
	return weightedPick(tied)
}

func durationFromEnv(name string, fallback time.Duration) time.Duration {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(raw)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

func intFromEnv(name string, fallback int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(raw)
	if err != nil || parsed < 0 {
		return fallback
	}
	return parsed
}

// sessionKeyFromContext reads the client's session identity off the request headers. A client
// that sends none is not sticky, which is the right default: one-shot calls do not care, and
// guessing an identity from the body would be a much larger claim.
func sessionKeyFromContext(ctx *schemas.BifrostContext) string {
	if ctx == nil {
		return ""
	}
	raw, ok := ctx.Value(schemas.BifrostContextKeyRequestHeaders).(map[string]string)
	if !ok {
		return ""
	}
	for _, header := range []string{"x-session-affinity", "x-session-id", "session_id", "x-client-request-id"} {
		if value, found := raw[header]; found && value != "" {
			return value
		}
	}
	return ""
}

// weightedPick spreads the choice across candidates in proportion to their weight.
func weightedPick(candidates []schemas.ProviderCandidate) schemas.ProviderCandidate {
	if len(candidates) == 0 {
		return schemas.ProviderCandidate{}
	}
	total := 0.0
	for _, candidate := range candidates {
		total += getWeight(candidate.Weight)
	}
	if total <= 0 {
		return candidates[0]
	}
	randomValue := rand.Float64() * total
	current := 0.0
	for _, candidate := range candidates {
		current += getWeight(candidate.Weight)
		if randomValue <= current {
			return candidate
		}
	}
	return candidates[len(candidates)-1]
}
