"""The outbound HTTP identity every fetch path in this repo sends.

One definition, imported by `market_backfill.py`, `build_site.py` and
`espn_403_probe.py`. It lives in its own module for two reasons: so the
value cannot drift between the modules that send it and the probe that
verifies it -- this repo has already paid for that once, see the workflow-pin
entry in CLAUDE.md -- and so the probe stays standard-library-only, which
importing `market_backfill` (pandas, numpy) would break.

Why it looks like this
----------------------
Measured, not guessed. On 2026-08-04, between 11:00Z and 18:56Z,
`site.api.espn.com`'s Akamai edge began answering 403 "Access Denied" to
requests carrying a `Mozilla/...` User-Agent unaccompanied by the rest of a
real browser's headers. That took out the pregame odds strip and stalled the
market backfill. `espn_403_probe.py` measured nine header sets, three
attempts each, from a GitHub runner:

    bare `Mozilla/5.0`          403  0/3   <- what market_backfill sent
    Chrome UA + Accept          403  0/3   <- what build_site sent
    Chrome UA + Accept + Lang   403  0/3
    Chrome UA + Lang + Referer  403  0/3
    urllib default              200  3/3
    requests default            200  3/3
    curl/8.5.0                  200  3/3
    full browser set            200  3/3
    this                        200  3/3

The discriminator is not "browser or not" but *consistent* or not: a bare
`Mozilla/5.0` claims to be a browser and brings none of a browser's other
headers, and that mismatch is what bot-management scores. Requests that never
claim to be a browser were not flagged at all. Only that one host was
blocking -- `sports.core.api.espn.com` answered all nine variants -- and
StatsAPI answered on the same egress, so this was never an outage or a
runner problem.

Two variants passed 3/3 and only one is maintainable. The "full browser set"
passes by impersonation, which means keeping a convincing Chrome fingerprint
current forever against a vendor actively scoring it, and losing the round
after next -- while making our traffic indistinguishable from the abuse the
rule exists to stop. An honest agent string says who we are and how to reach
us, which is what a rate-limit conversation needs, and it is the form
`schedule_gate.py` already uses against StatsAPI.

If ESPN 403s again, run the probe before changing this. A header set that
stopped working is a measurement, not a guess to iterate on.
"""

FETCH_HEADERS = {
    "User-Agent": "Dave356w-matchup-site/1.0 "
                  "(+https://github.com/Dave356w/Dave356w)",
    "Accept": "application/json",
}
