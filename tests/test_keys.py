"""Tests for Redis key/channel name conventions."""

from __future__ import annotations

from ahp.core.address import AgentAddress
from ahp.transport.keys import PREFIX, Keys


def test_all_keys_share_prefix():
    addr = AgentAddress.parse("o.r.d.sd.s.session.i")
    builders = [
        Keys.agent_channel(addr),
        Keys.reply_channel("abc"),
        Keys.thread_stream("t::x"),
        Keys.registry_hash(),
        Keys.alive_key(addr),
        Keys.cache_key("deadbeef"),
        Keys.cache_scan_pattern(),
    ]
    assert all(k.startswith(f"{PREFIX}:") for k in builders), builders


def test_agent_channel_accepts_address_or_string():
    addr = AgentAddress.parse("o.r.d.sd.s.session.i")
    assert Keys.agent_channel(addr) == Keys.agent_channel(str(addr))


def test_distinct_addresses_yield_distinct_channels():
    a = AgentAddress.parse("o.r.d.sd.s.session.alice")
    b = AgentAddress.parse("o.r.d.sd.s.session.bob")
    assert Keys.agent_channel(a) != Keys.agent_channel(b)


def test_scan_pattern_matches_cache_keys():
    import fnmatch
    pat = Keys.cache_scan_pattern()
    assert fnmatch.fnmatch(Keys.cache_key("abc123"), pat)
