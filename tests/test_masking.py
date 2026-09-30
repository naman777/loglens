import pytest

from loglens.tokenizer.masking import encode_line, mask


@pytest.mark.parametrize("raw,expected", [
    ("at 2005-06-03 15:42:50.363 done", "at <TS> done"),
    ("2015-07-29T17:41:41Z boot", "<TS> boot"),
    ("started 17:41:41,536 ok", "started <TS> ok"),
    ("Thu Jun 09 06:07:04 2005 restart", "<TS> restart"),
    ("Jun  9 06:06:20 combo", "<TS> combo"),
    ("date 2005.06.03 end", "date <TS> end"),
])
def test_timestamps(raw, expected):
    assert mask(raw) == expected


@pytest.mark.parametrize("raw", ["retry 3 of 5", "version v2.3", "ratio 1:2", "id abc:def"])
def test_timestamps_negative(raw):
    assert "<TS>" not in mask(raw)


@pytest.mark.parametrize("raw,expected", [
    ("req 38101a0b-2096-447d-96ea-a692162415ae ok", "req <UUID> ok"),
    ("req-9BC36DD9-91C5-4314-898A-F1FA9A9CBD5B x", "req-<UUID> x"),
    ("i 113d3a99-c3da-401f-bbb2-cc2caa5b9683", "i <UUID>"),
    ("[instance: 1e3a6d1b-1111-2222-3333-444455556666]", "[instance: <UUID>]"),
    ("uuid=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee;", "uuid=<UUID>;"),
])
def test_uuid(raw, expected):
    assert mask(raw) == expected


@pytest.mark.parametrize("raw", ["aaaaaaaa-bbbb-cccc", "12345-67", "a-b-c-d-e"])
def test_uuid_negative(raw):
    assert "<UUID>" not in mask(raw)


@pytest.mark.parametrize("raw,expected", [
    ("from 10.250.19.102 done", "from <IP> done"),
    ("src: /10.250.19.102:54106 dest", "src: /<IP>:<PORT> dest"),
    ("connect 192.168.0.1:8080 failed", "connect <IP>:<PORT> failed"),
    ("peer fe80::1ff:fe23:4567:890a lost", "peer <IP> lost"),
    ("loopback ::1 up", "loopback <IP> up"),
])
def test_ip(raw, expected):
    assert mask(raw) == expected


@pytest.mark.parametrize("raw", ["version 1.2.3", "ratio 1.5", "at 12:30"])
def test_ip_negative(raw):
    assert "<IP>" not in mask(raw)


@pytest.mark.parametrize("raw,expected", [
    ("open /var/log/app/server.log fail", "open <PATH>/server.log fail"),
    ("cd /mnt/hadoop/mapred/system/job_2008/job.jar", "cd <PATH>/job.jar"),
    ("GET https://example.com/a/b?c=1 200", "GET <URL> 200"),
    ("file /tmp/x", "file <PATH>"),
    ("mkdir /usr/local/lib/", "mkdir <PATH>"),
])
def test_paths_and_urls(raw, expected):
    assert mask(raw) == expected


@pytest.mark.parametrize("raw", ["ratio a/b", "and/or", "1/2 done"])
def test_paths_negative_short_words(raw):
    # slash-joined words are not absolute paths
    assert "<PATH>" not in mask(raw)


@pytest.mark.parametrize("raw,expected", [
    ("addr 0xdeadbeef bad", "addr <HEX> bad"),
    ("hash 9f86d081884c7d65 ok", "hash <HEX> ok"),
    ("ptr 0x7ffe12 fault", "ptr <HEX> fault"),
    ("sum deadbeef01 mismatch", "sum <HEX> mismatch"),
    ("crc 0X1F end", "crc <HEX> end"),
])
def test_hex(raw, expected):
    assert mask(raw) == expected


@pytest.mark.parametrize("raw", ["feedback loop", "abcdefab is a word", "12345678 items"])
def test_hex_negative(raw):
    assert "<HEX>" not in mask(raw)


@pytest.mark.parametrize("raw,expected", [
    ("Receiving block blk_-1608999687919862906 src", "Receiving block <BLK> src"),
    ("deleting blk_123 now", "deleting <BLK> now"),
    ("ok blk_9", "ok <BLK>"),
    ("task application_1445062781478_0011 done", "task <ID> done"),
    ("token Zbcdef0123456789Zbcdef01 bad", "token <ID> bad"),
])
def test_blk_and_ids(raw, expected):
    assert mask(raw) == expected


@pytest.mark.parametrize("raw", ["blk missing", "abcdefghijklmnopqrstuv", "ID 42"])
def test_ids_negative(raw):
    m = mask(raw)
    assert "<BLK>" not in m and "<ID>" not in m


@pytest.mark.parametrize("raw,expected", [
    ("took 5 ms", "took <DURATION:<10ms>"),
    ("took 50ms", "took <DURATION:<100ms>"),
    ("took 0.5 s", "took <DURATION:<1s>"),
    ("took 3 sec", "took <DURATION:<10s>"),
    ("waited 25s", "waited <DURATION:10s+>"),
    ("waited 2 min", "waited <DURATION:10s+>"),
])
def test_duration_buckets(raw, expected):
    assert mask(raw) == expected


@pytest.mark.parametrize("raw", ["5 items", "abc5s", "3 sessions"])
def test_duration_negative(raw):
    assert "<DURATION" not in mask(raw)


@pytest.mark.parametrize("raw,expected", [
    ("retry 3 times", "retry <NUM> times"),
    ("size 12345 bytes", "size <NUM> bytes"),
    ("load 0.75", "load <NUM>"),
    ("status 503 upstream", "status 503 upstream"),
    ('"GET /x HTTP/1.1" 200 512', '"GET <PATH> HTTP/<NUM>" 200 <NUM>'),
    ("exit code 137", "exit code 137"),
    ("worker-3 up", "worker-<NUM> up"),
    ("45% used", "<NUM>% used"),
])
def test_numbers(raw, expected):
    assert mask(raw) == expected


def test_version_strings_and_dates_not_shredded():
    assert mask("running v2.3.1 build") == "running <VER> build"
    assert mask("kernel 2.6.32-5-amd64") == "kernel <VER>"
    assert mask("released 2015-07-29") == "released <TS>"
    assert mask("release v2 stable") == "release v2 stable"
    assert mask("ver 10.4.1-beta") == "ver <VER>"


def test_bgl_location():
    assert mask("node R02-M1-N0-C:J12-U11 died") == "node <LOC> died"


def test_masking_is_idempotent():
    s = "conn 10.0.0.1:80 took 5ms blk_1 /a/b.log at 12:00:01 status 500"
    assert mask(mask(s)) == mask(s)


def test_structural_prefix_tokens():
    out = encode_line("disk full", level="warning", service="worker", known_services={"worker"})
    assert out == "<LVL:WARN> <SVC:worker> disk full"
    out = encode_line("x", level=None, service="mystery", known_services={"worker"})
    assert out.startswith("<LVL:UNK> <SVC:other>")


def test_ipv4_mapped_and_relative_paths():
    assert mask("from ::ffff:10.1.2.3 port 22") == "from <IP> port <NUM>"
    assert mask("cd mnt_projects/sysapps/src/ib file") == "cd <PATH> file"
    assert mask("[/mnt/a/b-3.2.0/x.c:88]") == "[<PATH>/x.c:<NUM>]"


@pytest.mark.parametrize("raw,expected", [
    ("generating core.21370", "generating core.<NUM>"),
    ("cache step_12 done", "cache step_<NUM> done"),
    ("Lustre mount FAILED : bglio1023 : point", "Lustre mount FAILED : bglio<NUM> : point"),
    ("device sda1 mounted", "device sda<NUM> mounted"),
    ("Setting hostname an7: ok", "Setting hostname an<NUM>: ok"),
    ("using ssh2 and sha256 with utf8", "using ssh2 and sha256 with utf8"),
    ("arch x86_64 ipv4 md5", "arch x86_64 ipv4 md5"),
])
def test_identifier_suffix_numbers(raw, expected):
    assert mask(raw) == expected


def test_fast_mask_equals_reference():
    from loglens.tokenizer.masking import mask_reference

    samples = [
        "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "took 5 ms", "no digits here at all", "path /a/b/c.txt and http://x.y/z?q=1", "R02-M1-N0-C:J12-U11 core.123",
        "0xdeadbeef and 9f86d081884c7d65 x86_64 utf8 sda1 ::1 fe80::1", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "status 503 code=137 v2.3.1 1.2.3.4:80 step_5", "", "   ", "a::b std::string x-y-z",
    ]
    for m in samples:
        assert mask(m) == mask_reference(m), m
