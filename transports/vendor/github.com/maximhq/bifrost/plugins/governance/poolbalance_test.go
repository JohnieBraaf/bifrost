package governance

import (
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

func candidates(names ...string) []schemas.ProviderCandidate {
	out := make([]schemas.ProviderCandidate, 0, len(names))
	weight := 1.0
	for _, name := range names {
		out = append(out, schemas.ProviderCandidate{Provider: name, Weight: &weight})
	}
	return out
}

// newTestBalancer builds a balancer with a controllable clock so the session TTL can be exercised
// without sleeping.
func newTestBalancer(gap int, ttl time.Duration) (*poolBalancer, *time.Time) {
	now := time.Date(2026, 11, 23, 12, 0, 0, 0, time.UTC)
	balancer := &poolBalancer{
		sessions:     make(map[string]*poolSession),
		ttl:          ttl,
		rebalanceGap: gap,
		now:          func() time.Time { return now },
	}
	return balancer, &now
}

// TestPoolBalancer_SpreadsNewSessions is the property the pool exists for: two sessions must not
// land on the same provider while the other one is idle.
func TestPoolBalancer_SpreadsNewSessions(t *testing.T) {
	balancer, _ := newTestBalancer(2, time.Hour)
	pool := candidates("a", "b")

	first, _ := balancer.pick("session-1", pool)
	second, _ := balancer.pick("session-2", pool)

	require.NotEqual(t, first.Provider, second.Provider)
}

func TestPoolBalancer_KeepsASessionPut(t *testing.T) {
	balancer, _ := newTestBalancer(2, time.Hour)
	pool := candidates("a", "b")

	first, _ := balancer.pick("session-1", pool)
	for i := 0; i < 5; i++ {
		again, reason := balancer.pick("session-1", pool)
		require.Equal(t, first.Provider, again.Provider, "a warm session must not move: %s", reason)
	}
}

func TestPoolBalancer_RebalancesOffAnOverloadedProvider(t *testing.T) {
	balancer, _ := newTestBalancer(2, time.Hour)
	pool := candidates("a", "b")

	// Put three sessions on "a" by hand, so "a" is two ahead of "b".
	now := balancer.now()
	balancer.sessions["s1"] = &poolSession{provider: "a", lastSeen: now}
	balancer.sessions["s2"] = &poolSession{provider: "a", lastSeen: now}
	balancer.sessions["s3"] = &poolSession{provider: "a", lastSeen: now}

	moved, reason := balancer.pick("s1", pool)
	require.Equal(t, "b", moved.Provider, "%s", reason)

	// The move is remembered, so the session does not bounce back on its next turn.
	again, _ := balancer.pick("s1", pool)
	require.Equal(t, "b", again.Provider)
}

func TestPoolBalancer_DoesNotRebalanceBelowTheGap(t *testing.T) {
	balancer, _ := newTestBalancer(2, time.Hour)
	pool := candidates("a", "b")
	now := balancer.now()
	balancer.sessions["s1"] = &poolSession{provider: "a", lastSeen: now}
	balancer.sessions["s2"] = &poolSession{provider: "a", lastSeen: now}
	balancer.sessions["s3"] = &poolSession{provider: "b", lastSeen: now}

	kept, reason := balancer.pick("s1", pool)
	require.Equal(t, "a", kept.Provider, "%s", reason)
}

func TestPoolBalancer_GapZeroNeverMovesAStickySession(t *testing.T) {
	balancer, _ := newTestBalancer(0, time.Hour)
	pool := candidates("a", "b")
	now := balancer.now()
	for _, key := range []string{"s1", "s2", "s3"} {
		balancer.sessions[key] = &poolSession{provider: "a", lastSeen: now}
	}

	kept, _ := balancer.pick("s1", pool)
	require.Equal(t, "a", kept.Provider)
}

func TestPoolBalancer_ExpiredSessionsFreeTheirProvider(t *testing.T) {
	balancer, now := newTestBalancer(2, 10*time.Minute)
	pool := candidates("a", "b")

	balancer.pick("session-1", pool) // lands somewhere
	require.Len(t, balancer.sessions, 1)

	*now = now.Add(11 * time.Minute)
	fresh, reason := balancer.pick("session-2", pool)

	// The expired session no longer counts, so the new one is free to take either provider; what
	// matters is that the expired entry is gone and the answer is a real candidate.
	require.Contains(t, []string{"a", "b"}, fresh.Provider, "%s", reason)
	require.Len(t, balancer.sessions, 1)

}

func TestPoolBalancer_ReassignsWhenTheProviderDropsOutOfThePool(t *testing.T) {
	balancer, _ := newTestBalancer(2, time.Hour)
	balancer.sessions["s1"] = &poolSession{provider: "gone", lastSeen: balancer.now()}

	chosen, reason := balancer.pick("s1", candidates("a", "b"))
	require.Contains(t, []string{"a", "b"}, chosen.Provider, "%s", reason)
	require.Equal(t, chosen.Provider, balancer.sessions["s1"].provider)
}

func TestPoolBalancer_WithoutASessionIdStaysWeighted(t *testing.T) {
	balancer, _ := newTestBalancer(2, time.Hour)
	pool := candidates("a", "b")

	seen := map[string]bool{}
	for i := 0; i < 200; i++ {
		chosen, reason := balancer.pick("", pool)
		require.Equal(t, "no session id: weighted pick", reason)
		seen[chosen.Provider] = true
	}
	require.True(t, seen["a"] && seen["b"], "weighted pick should not collapse onto one provider")
	require.Empty(t, balancer.sessions, "anonymous requests must not accumulate session state")
}

func TestPoolBalancer_SingleCandidateSkipsTheMap(t *testing.T) {
	balancer, _ := newTestBalancer(2, time.Hour)
	chosen, reason := balancer.pick("session-1", candidates("only"))
	require.Equal(t, "only", chosen.Provider)
	require.Equal(t, "only one candidate", reason)
	require.Empty(t, balancer.sessions)
}

// TestSessionKeyFromContext pins the header precedence: the affinity header pi sends wins, and a
// request with no headers is anonymous rather than an error.
func TestSessionKeyFromContext(t *testing.T) {
	ctx := schemas.NewBifrostContext(nil, time.Time{})
	require.Equal(t, "", sessionKeyFromContext(ctx))

	ctx.SetValue(schemas.BifrostContextKeyRequestHeaders, map[string]string{
		"x-client-request-id": "client-1",
		"x-session-affinity":  "session-1",
	})
	require.Equal(t, "session-1", sessionKeyFromContext(ctx))

	ctx.SetValue(schemas.BifrostContextKeyRequestHeaders, map[string]string{"x-session-id": "session-2"})
	require.Equal(t, "session-2", sessionKeyFromContext(ctx))
}
