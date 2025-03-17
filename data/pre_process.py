import argparse
import json
import os
import pandas as pd
from pathlib import Path
import re

CPU_HEADERS = [
    'user',  # Time spent in user mode
    'nice',  # Time spent in user mode with nice priority
    'system',  # Time spent in kernel mode
    'idle',  # Time spent idle
    'iowait',  # Time waiting for I/O
    'irq',  # Time spent servicing interrupts
    'softirq',  # Time spent servicing softirqs
    'steal',  # Time stolen by a hypervisor (for VMs)
    'guest',  # Time spent running virtual CPUs
    'guest_nice'  # Time running virtual CPUs with a nice priority
]

MEM_HEADERS = [
    'MemTotal',
    'MemAvailable',
    'MemFree',
    'Buffers',
    'Cached',
    'SwapTotal',
    'SwapFree',
    'SwapCached',
    'Dirty',
    'Writeback',
    'AnonPages',
    'Mapped',
    'Active(file)',
    'Inactive(file)',
    'KernelStack',
    'PageTables',
    'HugePages_Total',
    'DirectMap4k',
    'DirectMap2M',
    'DirectMap1G',
]

DISK_HEADER = [
    'device_name',
    'reads_completed',
    'reads_merged',
    'sectors_read',
    'time_reading_ms',
    'writes_completed',
    'writes_merged',
    'sectors_written',
    'time_writing_ms',
    'ios_in_progress',
    'time_ios_ms',
    'weighted_time_ios_ms',
]

RD = 0x1000
WR = 0x2000
RMW = 0x3000
SUB = 0x4000
CACHE = 0x8000

DATA = 0x0200
ATTR = 0x0300
EXEC = 0x0400
PG = 0x0500

OSD_OPS = {
    #  read
    RD | DATA | 1: ("read", "CEPH_OSD_OP_READ"),
    RD | DATA | 2: ("stat", "CEPH_OSD_OP_STAT"),
    RD | DATA | 3: ("mapext", "CEPH_OSD_OP_MAPEXT"),
    RD | DATA | 31: ("checksum", "CEPH_OSD_OP_CHECKSUM"),

    #  fancy read
    RD | DATA | 4: ("masktrunc", "CEPH_OSD_OP_MASKTRUNC"),
    RD | DATA | 5: ("sparse-read", "CEPH_OSD_OP_SPARSE_READ"),

    RD | DATA | 6: ("notify", "CEPH_OSD_OP_NOTIFY"),
    RD | DATA | 7: ("notify-ack", "CEPH_OSD_OP_NOTIFY_ACK"),

    #  versioning
    RD | DATA | 8: ("assert-version", "CEPH_OSD_OP_ASSERT_VER"),

    RD | DATA | 9: ("list-watchers", "CEPH_OSD_OP_LIST_WATCHERS"),

    RD | DATA | 10: ("list-snaps", "CEPH_OSD_OP_LIST_SNAPS"),

    #  sync
    RD | DATA | 11: ("sync_read", "CEPH_OSD_OP_SYNC_READ"),

    #  write
    WR | DATA | 1: ("write", "CEPH_OSD_OP_WRITE"),
    WR | DATA | 2: ("writefull", "CEPH_OSD_OP_WRITEFULL"),
    WR | DATA | 3: ("truncate", "CEPH_OSD_OP_TRUNCATE"),
    WR | DATA | 4: ("zero", "CEPH_OSD_OP_ZERO"),
    WR | DATA | 5: ("delete", "CEPH_OSD_OP_DELETE"),

    #  fancy write
    WR | DATA | 6: ("append", "CEPH_OSD_OP_APPEND"),
    WR | DATA | 7: ("startsync", "CEPH_OSD_OP_STARTSYNC"),
    WR | DATA | 8: ("settrunc", "CEPH_OSD_OP_SETTRUNC"),
    WR | DATA | 9: ("trimtrunc", "CEPH_OSD_OP_TRIMTRUNC"),

    RMW | DATA | 10: ("tmapup", "CEPH_OSD_OP_TMAPUP"),
    WR | DATA | 11: ("tmapput", "CEPH_OSD_OP_TMAPPUT"),
    RD | DATA | 12: ("tmapget", "CEPH_OSD_OP_TMAPGET"),

    WR | DATA | 13: ("create", "CEPH_OSD_OP_CREATE"),
    WR | DATA | 14: ("rollback", "CEPH_OSD_OP_ROLLBACK"),

    WR | DATA | 15: ("watch", "CEPH_OSD_OP_WATCH"),

    #  omap
    RD | DATA | 17: ("omap-get-keys", "CEPH_OSD_OP_OMAPGETKEYS"),
    RD | DATA | 18: ("omap-get-vals", "CEPH_OSD_OP_OMAPGETVALS"),
    RD | DATA | 19: ("omap-get-header", "CEPH_OSD_OP_OMAPGETHEADER"),
    RD | DATA | 20: ("omap-get-vals-by-keys", "CEPH_OSD_OP_OMAPGETVALSBYKEYS"),
    WR | DATA | 21: ("omap-set-vals", "CEPH_OSD_OP_OMAPSETVALS"),
    WR | DATA | 22: ("omap-set-header", "CEPH_OSD_OP_OMAPSETHEADER"),
    WR | DATA | 23: ("omap-clear", "CEPH_OSD_OP_OMAPCLEAR"),
    WR | DATA | 24: ("omap-rm-keys", "CEPH_OSD_OP_OMAPRMKEYS"),
    WR | DATA | 44: ("omap-rm-key-range", "CEPH_OSD_OP_OMAPRMKEYRANGE"),
    RD | DATA | 25: ("omap-cmp", "CEPH_OSD_OP_OMAP_CMP"),

    #  tiering
    WR | DATA | 26: ("copy-from", "CEPH_OSD_OP_COPY_FROM"),
    WR | DATA | 45: ("copy-from2", "CEPH_OSD_OP_COPY_FROM2"),
    #  was copy-get-classic
    WR | DATA | 28: ("undirty", "CEPH_OSD_OP_UNDIRTY"),
    RD | DATA | 29: ("isdirty", "CEPH_OSD_OP_ISDIRTY"),
    RD | DATA | 30: ("copy-get", "CEPH_OSD_OP_COPY_GET"),
    CACHE | DATA | 31: ("cache-flush", "CEPH_OSD_OP_CACHE_FLUSH"),
    CACHE | DATA | 32: ("cache-evict", "CEPH_OSD_OP_CACHE_EVICT"),
    CACHE | DATA | 33: ("cache-try-flush", "CEPH_OSD_OP_CACHE_TRY_FLUSH"),

    #  convert tmap to omap
    RMW | DATA | 34: ("tmap2omap", "CEPH_OSD_OP_TMAP2OMAP"),

    #  hints
    WR | DATA | 35: ("set-alloc-hint", "CEPH_OSD_OP_SETALLOCHINT"),

    #  cache pin/unpin
    WR | DATA | 36: ("cache-pin", "CEPH_OSD_OP_CACHE_PIN"),
    WR | DATA | 37: ("cache-unpin", "CEPH_OSD_OP_CACHE_UNPIN"),

    #  ESX/SCSI
    WR | DATA | 38: ("write-same", "CEPH_OSD_OP_WRITESAME"),
    RD | DATA | 32: ("cmpext", "CEPH_OSD_OP_CMPEXT"),

    #  Extensible
    WR | DATA | 39: ("set-redirect", "CEPH_OSD_OP_SET_REDIRECT"),
    CACHE | DATA | 40: ("set-chunk", "CEPH_OSD_OP_SET_CHUNK"),
    WR | DATA | 41: ("tier-promote", "CEPH_OSD_OP_TIER_PROMOTE"),
    WR | DATA | 42: ("unset-manifest", "CEPH_OSD_OP_UNSET_MANIFEST"),
    CACHE | DATA | 43: ("tier-flush", "CEPH_OSD_OP_TIER_FLUSH"),
    CACHE | DATA | 44: ("tier-evict", "CEPH_OSD_OP_TIER_EVICT"),

    # * attrs *
    #  read
    RD | ATTR | 1: ("getxattr", "CEPH_OSD_OP_GETXATTR"),
    RD | ATTR | 2: ("getxattrs", "CEPH_OSD_OP_GETXATTRS"),
    RD | ATTR | 3: ("cmpxattr", "CEPH_OSD_OP_CMPXATTR"),

    #  write
    WR | ATTR | 1: ("setxattr", "CEPH_OSD_OP_SETXATTR"),
    WR | ATTR | 2: ("setxattrs", "CEPH_OSD_OP_SETXATTRS"),
    WR | ATTR | 3: ("resetxattrs", "CEPH_OSD_OP_RESETXATTRS"),
    WR | ATTR | 4: ("rmxattr", "CEPH_OSD_OP_RMXATTR"),

    # * subop *
    SUB | 1: ("pull", "CEPH_OSD_OP_PULL"),
    SUB | 2: ("push", "CEPH_OSD_OP_PUSH"),
    SUB | 3: ("balance-reads", "CEPH_OSD_OP_BALANCEREADS"),
    SUB | 4: ("unbalance-reads", "CEPH_OSD_OP_UNBALANCEREADS"),
    SUB | 5: ("scrub", "CEPH_OSD_OP_SCRUB"),
    SUB | 6: ("scrub-reserve", "CEPH_OSD_OP_SCRUB_RESERVE"),
    SUB | 7: ("scrub-unreserve", "CEPH_OSD_OP_SCRUB_UNRESERVE"),
    #  8 used to be scrub-stop
    SUB | 9: ("scrub-map", "CEPH_OSD_OP_SCRUB_MAP"),

    # * exec *
    #  note: the RD bit here is wrong; see special-case below in helper
    RD | EXEC | 1: ("call", "CEPH_OSD_OP_CALL"),

    # * pg *
    RD | PG | 1: ("pgls", "CEPH_OSD_OP_PGLS"),
    RD | PG | 2: ("pgls-filter", "CEPH_OSD_OP_PGLS_FILTER"),
    RD | PG | 3: ("pg-hitset-ls", "CEPH_OSD_OP_PG_HITSET_LS"),
    RD | PG | 4: ("pg-hitset-get", "CEPH_OSD_OP_PG_HITSET_GET"),
    RD | PG | 5: ("pgnls", "CEPH_OSD_OP_PGNLS"),
    RD | PG | 6: ("pgnls-filter", "CEPH_OSD_OP_PGNLS_FILTER"),
    RD | PG | 7: ("scrubls", "CEPH_OSD_OP_SCRUBLS"),
}

for idx, code in enumerate(sorted(list(OSD_OPS.keys()))):
    osd_op_name, osd_op_type = OSD_OPS[code]
    OSD_OPS[code] = (osd_op_name, osd_op_type, idx)

IDX_TO_OSD_OPS = {}
for code, (osd_op_name, osd_op_type, idx) in OSD_OPS.items():
    IDX_TO_OSD_OPS[idx] = {'op_code': code, 'name': osd_op_name, 'type': osd_op_type}

MSG_OSD_OPS = {
    # misc
    1: "CEPH_MSG_SHUTDOWN",
    2: "CEPH_MSG_PING",

    # client <-> monitor
    4: "CEPH_MSG_MON_MAP",
    5: "CEPH_MSG_MON_GET_MAP",
    6: "CEPH_MSG_MON_GET_OSDMAP",
    7: "CEPH_MSG_MON_METADATA",
    13: "CEPH_MSG_STATFS",
    14: "CEPH_MSG_STATFS_REPLY",
    15: "CEPH_MSG_MON_SUBSCRIBE",
    16: "CEPH_MSG_MON_SUBSCRIBE_ACK",
    17: "CEPH_MSG_AUTH",
    18: "CEPH_MSG_AUTH_REPLY",
    19: "CEPH_MSG_MON_GET_VERSION",
    20: "CEPH_MSG_MON_GET_VERSION_REPLY",

    # client <-> mds
    21: "CEPH_MSG_MDS_MAP",

    22: "CEPH_MSG_CLIENT_SESSION",
    23: "CEPH_MSG_CLIENT_RECONNECT",

    24: "CEPH_MSG_CLIENT_REQUEST",
    25: "CEPH_MSG_CLIENT_REQUEST_FORWARD",
    26: "CEPH_MSG_CLIENT_REPLY",
    27: "CEPH_MSG_CLIENT_RECLAIM",
    28: "CEPH_MSG_CLIENT_RECLAIM_REPLY",
    29: "CEPH_MSG_CLIENT_METRICS",
    0x310: "CEPH_MSG_CLIENT_CAPS",
    0x311: "CEPH_MSG_CLIENT_LEASE",
    0x312: "CEPH_MSG_CLIENT_SNAP",
    0x313: "CEPH_MSG_CLIENT_CAPRELEASE",
    0x314: "CEPH_MSG_CLIENT_QUOTA",

    # pool ops
    48: "CEPH_MSG_POOLOP_REPLY",
    49: "CEPH_MSG_POOLOP",

    # osd
    41: "CEPH_MSG_OSD_MAP",
    42: "CEPH_MSG_OSD_OP",
    43: "CEPH_MSG_OSD_OPREPLY",
    44: "CEPH_MSG_WATCH_NOTIFY",
    61: "CEPH_MSG_OSD_BACKOFF",

    # FSMap subscribers (see all MDS clusters at once)
    45: "CEPH_MSG_FS_MAP",
    # FSMapUser subscribers (get MDS clusters name->ID mapping)
    103: "CEPH_MSG_FS_MAP_USER",
    # monitor internal
    64: "MSG_MON_SCRUB",
    65: "MSG_MON_ELECTION",
    66: "MSG_MON_PAXOS",
    67: "MSG_MON_PROBE",
    68: "MSG_MON_JOIN",
    69: "MSG_MON_SYNC",
    140: "MSG_MON_PING",

    # monitor <-> mon admin tool
    50: "MSG_MON_COMMAND",
    51: "MSG_MON_COMMAND_ACK",
    52: "MSG_LOG",
    53: "MSG_LOGACK",

    58: "MSG_GETPOOLSTATS",
    59: "MSG_GETPOOLSTATSREPLY",

    60: "MSG_MON_GLOBAL_ID",
    141: "MSG_MON_USED_PENDING_KEYS",

    47: "MSG_ROUTE",
    46: "MSG_FORWARD",

    40: "MSG_PAXOS",

    62: "MSG_CONFIG",
    63: "MSG_GET_CONFIG",

    54: "MSG_KV_DATA",

    76: "MSG_MON_GET_PURGED_SNAPS",
    77: "MSG_MON_GET_PURGED_SNAPS_REPLY",

    # osd internal
    70: "MSG_OSD_PING",
    71: "MSG_OSD_BOOT",
    72: "MSG_OSD_FAILURE",
    73: "MSG_OSD_ALIVE",
    74: "MSG_OSD_MARK_ME_DOWN",
    75: "MSG_OSD_FULL",
    123: "MSG_OSD_MARK_ME_DEAD",

    78: "MSG_OSD_PGTEMP",

    79: "MSG_OSD_BEACON",

    80: "MSG_OSD_PG_NOTIFY",
    130: "MSG_OSD_PG_NOTIFY2",
    81: "MSG_OSD_PG_QUERY",
    131: "MSG_OSD_PG_QUERY2",
    83: "MSG_OSD_PG_LOG",
    84: "MSG_OSD_PG_REMOVE",
    85: "MSG_OSD_PG_INFO",
    132: "MSG_OSD_PG_INFO2",
    86: "MSG_OSD_PG_TRIM",

    87: "MSG_PGSTATS",
    88: "MSG_PGSTATSACK",

    89: "MSG_OSD_PG_CREATE",
    90: "MSG_REMOVE_SNAPS",

    91: "MSG_OSD_SCRUB",
    92: "MSG_OSD_SCRUB_RESERVE",  # previous PG_MISSING
    93: "MSG_OSD_REP_SCRUB",

    94: "MSG_OSD_PG_SCAN",
    95: "MSG_OSD_PG_BACKFILL",
    96: "MSG_OSD_PG_BACKFILL_REMOVE",

    97: "MSG_COMMAND",
    98: "MSG_COMMAND_REPLY",

    99: "MSG_OSD_BACKFILL_RESERVE",
    150: "MSG_OSD_RECOVERY_RESERVE",
    151: "MSG_OSD_FORCE_RECOVERY",

    105: "MSG_OSD_PG_PUSH",
    106: "MSG_OSD_PG_PULL",
    107: "MSG_OSD_PG_PUSH_REPLY",

    108: "MSG_OSD_EC_WRITE",
    109: "MSG_OSD_EC_WRITE_REPLY",
    110: "MSG_OSD_EC_READ",
    111: "MSG_OSD_EC_READ_REPLY",

    112: "MSG_OSD_REPOP",
    113: "MSG_OSD_REPOPREPLY",
    114: "MSG_OSD_PG_UPDATE_LOG_MISSING",
    115: "MSG_OSD_PG_UPDATE_LOG_MISSING_REPLY",

    136: "MSG_OSD_PG_PCT",

    116: "MSG_OSD_PG_CREATED",
    117: "MSG_OSD_REP_SCRUBMAP",
    118: "MSG_OSD_PG_RECOVERY_DELETE",
    119: "MSG_OSD_PG_RECOVERY_DELETE_REPLY",
    120: "MSG_OSD_PG_CREATE2",
    121: "MSG_OSD_SCRUB2",

    122: "MSG_OSD_PG_READY_TO_MERGE",

    133: "MSG_OSD_PG_LEASE",
    134: "MSG_OSD_PG_LEASE_ACK",

    # MDS

    100: "MSG_MDS_BEACON",  # to monitor
    101: "MSG_MDS_PEER_REQUEST",
    102: "MSG_MDS_TABLE_REQUEST",
    135: "MSG_MDS_SCRUB",

    0x200: "MSG_MDS_RESOLVE",  # 0x2xx are for mdcache of mds
    0x201: "MSG_MDS_RESOLVEACK",
    0x202: "MSG_MDS_CACHEREJOIN",
    0x203: "MSG_MDS_DISCOVER",
    0x204: "MSG_MDS_DISCOVERREPLY",
    0x205: "MSG_MDS_INODEUPDATE",
    0x206: "MSG_MDS_DIRUPDATE",
    0x207: "MSG_MDS_CACHEEXPIRE",
    0x208: "MSG_MDS_DENTRYUNLINK",
    0x209: "MSG_MDS_FRAGMENTNOTIFY",
    0x20a: "MSG_MDS_OFFLOAD_TARGETS",
    0x20c: "MSG_MDS_DENTRYLINK",
    0x20d: "MSG_MDS_FINDINO",
    0x20e: "MSG_MDS_FINDINOREPLY",
    0x20f: "MSG_MDS_OPENINO",
    0x210: "MSG_MDS_OPENINOREPLY",
    0x211: "MSG_MDS_SNAPUPDATE",
    0x212: "MSG_MDS_FRAGMENTNOTIFYACK",
    0x300: "MSG_MDS_LOCK",  # 0x3xx are for locker of mds
    0x301: "MSG_MDS_INODEFILECAPS",

    0x449: "MSG_MDS_EXPORTDIRDISCOVER",  # 0x4xx are for migrator of mds
    0x450: "MSG_MDS_EXPORTDIRDISCOVERACK",
    0x451: "MSG_MDS_EXPORTDIRCANCEL",
    0x452: "MSG_MDS_EXPORTDIRPREP",
    0x453: "MSG_MDS_EXPORTDIRPREPACK",
    0x454: "MSG_MDS_EXPORTDIRWARNING",
    0x455: "MSG_MDS_EXPORTDIRWARNINGACK",
    0x456: "MSG_MDS_EXPORTDIR",
    0x457: "MSG_MDS_EXPORTDIRACK",
    0x458: "MSG_MDS_EXPORTDIRNOTIFY",
    0x459: "MSG_MDS_EXPORTDIRNOTIFYACK",
    0x460: "MSG_MDS_EXPORTDIRFINISH",

    0x470: "MSG_MDS_EXPORTCAPS",
    0x471: "MSG_MDS_EXPORTCAPSACK",
    0x472: "MSG_MDS_GATHERCAPS",

    0x500: "MSG_MDS_HEARTBEAT",  # for mds load balancer
    0x501: "MSG_MDS_METRICS",  # for mds metric aggregator
    0x502: "MSG_MDS_PING",  # for mds pinger
    0x503: "MSG_MDS_SCRUB_STATS",  # for mds scrub stack
    0x505: "MSG_MDS_QUIESCE_DB_LISTING",  # quiesce db replication
    0x506: "MSG_MDS_QUIESCE_DB_ACK",  # quiesce agent ack back to the db

    # generic
    0x600: "MSG_TIMECHECK",
    0x601: "MSG_MON_HEALTH",

    # Message::encode() crcflags bits
    1 << 0: "MSG_CRC_DATA",
    1 << 1: "MSG_CRC_HEADER",
    (1 << 0) | (1 << 1): "MSG_CRC_ALL",

    # Special
    0x607: "MSG_NOP",

    0x608: "MSG_MON_HEALTH_CHECKS",
    0x609: "MSG_TIMECHECK2",

    # ceph-mgr <-> OSD/MDS daemons
    0x700: "MSG_MGR_OPEN",
    0x701: "MSG_MGR_CONFIGURE",
    0x702: "MSG_MGR_REPORT",

    # ceph-mgr <-> ceph-mon
    0x703: "MSG_MGR_BEACON",

    # ceph-mon(MgrMonitor) -> OSD/MDS daemons
    0x704: "MSG_MGR_MAP",

    # ceph-mon(MgrMonitor) -> ceph-mgr
    0x705: "MSG_MGR_DIGEST",
    # cephmgr -> ceph-mon
    0x706: "MSG_MON_MGR_REPORT",
    0x707: "MSG_SERVICE_MAP",

    0x708: "MSG_MGR_CLOSE",
    0x709: "MSG_MGR_COMMAND",
    0x70a: "MSG_MGR_COMMAND_REPLY",

    # ceph-mgr <-> MON daemons
    0x70b: "MSG_MGR_UPDATE",

    # nvmeof mon -> gw daemons
    0x800: "MSG_MNVMEOF_GW_MAP",

    # gw daemons -> nvmeof mon
    0x801: "MSG_MNVMEOF_GW_BEACON",
}

for idx, code in enumerate(sorted(list(MSG_OSD_OPS.keys()))):
    op_type = MSG_OSD_OPS[code]
    MSG_OSD_OPS[code] = (op_type, idx)

IDX_TO_MSG_OSD_OPS = {}
for code, (op_type, idx) in MSG_OSD_OPS.items():
    IDX_TO_MSG_OSD_OPS[idx] = {'op_code': code, 'type': op_type}


def read_cpu_info(path):
    cpu_info = {}
    try:
        with open(path, 'r') as file:
            lines = file.readlines()
            total_cpu_line = lines[0].split()
            cpu_count = len(list(filter(lambda line: re.match(r'^cpu\d+.*', line), lines)))
            for header, val in zip(CPU_HEADERS, total_cpu_line[1:]):
                cpu_info[header] = int(val)
            cpu_info['cpu_count'] = cpu_count
    except Exception as ex:
        print(f"Error reading {path}: {ex}")
    return cpu_info


def read_mem_info(path):
    mem_info = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    if key not in MEM_HEADERS:
                        continue
                    value = parts[1].strip().split()[0]
                    mem_info[key] = int(value)
    except Exception as ex:
        print(f"Error reading {path}: {ex}")
    return mem_info


def read_disk_info(path, disk_labels):
    disk_info = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()[2:]
                if len(parts) >= len(DISK_HEADER) and parts[0] in disk_labels:
                    parts = parts[:len(DISK_HEADER)]
                    data = {DISK_HEADER[i]: int(parts[i]) if i > 2 else parts[i] for i in range(len(parts))}
                    disk_info[disk_labels[data['device_name']]] = data
    except Exception as ex:
        print(f"Error reading {path}: {ex}")
    return disk_info


def read_system_state(path, disk_labels):
    cpu_path = os.path.join(path, 'cpu.txt')
    mem_path = os.path.join(path, 'mem.txt')
    disk_path = os.path.join(path, 'disk_stats.txt')
    return read_cpu_info(cpu_path), read_mem_info(mem_path), read_disk_info(disk_path, disk_labels)


def read_disk_labels(path):
    disk_labels = {}
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) < 11:
                continue
            if re.match(r'osd-device-\d+-\w+', parts[8]):
                disk_labels[os.path.basename(parts[10])] = parts[8]
    return disk_labels


def read_osd_data(osd_data_path):
    disk_labels = read_disk_labels(os.path.join(osd_data_path, 'disks_labels.txt'))
    system_states = []
    entries = None
    ops = None
    for item in Path(osd_data_path).iterdir():
        if item.is_file() and re.match(r'^entries_.*\.csv$', item.name):
            entries = pd.read_csv(item)
        if item.is_file() and re.match(r'^ops_.*\.csv$', item.name):
            ops = pd.read_csv(item)
        if item.is_dir() and re.match(r'^\d+$', item.name):
            time = int(item.name)
            system_state = {'timestamp': time}
            cpu_info, mem_info, disk_info = read_system_state(item, disk_labels)
            for field, val in cpu_info.items():
                system_state[f'cpu_{field}'] = val
            for field, val in mem_info.items():
                system_state[f'mem_{field}'] = val
            for partition, partition_data in disk_info.items():
                partition = partition.replace('-', '_')
                for field, val in partition_data.items():
                    system_state[f'disk_{partition}_{field}'] = val
            system_states.append(system_state)
    return entries, ops, pd.DataFrame(system_states)


def read_experiment_data(exp_path):
    exp_data = {}
    for item in Path(exp_path).iterdir():
        if re.match(r'^data.osd\d+$', item.name):
            osd_name = item.name.split('.')[1]
            entries, ops, system_states = read_osd_data(item)
            exp_data[osd_name] = {
                'entries': entries,
                'ops': ops,
                'system_states': system_states
            }
    return exp_data


def store_exp_data(exp_data, output_path):
    for osd_name, osd_data in exp_data.items():
        osd_output_path = os.path.join(output_path, osd_name)
        if not os.path.exists(osd_output_path):
            os.makedirs(osd_output_path)
        for name, data in osd_data.items():
            if isinstance(data, pd.DataFrame):
                path = os.path.join(osd_output_path, f'{name}.csv')
                data.to_csv(path, index=False)
            else:
                print(f'Unknown data type for {name}: {type(data)}. We expect pandas.DataFrame.')
        idx_to_osd_op_path = os.path.join(osd_output_path, 'osd_op_types.json')
        with open(idx_to_osd_op_path, "w") as file:
            json.dump(IDX_TO_OSD_OPS, file)
        idx_to_msg_op_path = os.path.join(osd_output_path, 'msg_op_types.json')
        with open(idx_to_msg_op_path, "w") as file:
            json.dump(IDX_TO_MSG_OSD_OPS, file)


def preprocess_system_states(data_dict: dict):
    for osd_name, osd_data in data_dict.items():
        state_df = osd_data['system_states']
        state_df.sort_values(by=['timestamp'], inplace=True)


def preprocess_entries(data_dict: dict):
    for osd_name, osd_data in data_dict.items():
        entries_df = osd_data['entries']
        entries_df['type'] = entries_df['type'].map(lambda x: MSG_OSD_OPS[x][1])
        entries_df['timestamp'] = entries_df['dequeue_stamp']
        entries_df['latency'] = entries_df['dequeue_end_stamp'] - entries_df['dequeue_stamp']
        entries_df.sort_values(by=['timestamp'], inplace=True)
        ops_df = osd_data['ops']
        ops_df['type'] = ops_df['type'].map(lambda x: OSD_OPS[x][2])


def read_all(path):
    data = {}
    for item in Path(path).iterdir():
        data_dict = read_experiment_data(item)
        for osd_name in data_dict.keys():
            if osd_name not in data:
                data[osd_name] = {}
            for data_name in data_dict[osd_name].keys():
                if data_name not in data[osd_name]:
                    data[osd_name][data_name] = []
                data[osd_name][data_name].append(data_dict[osd_name][data_name])
    for osd_name in data.keys():
        for data_name in data[osd_name].keys():
            data[osd_name][data_name] = pd.concat(data[osd_name][data_name])
    return data


def main(args):
    data_dict = read_all(args.input)
    preprocess_system_states(data_dict)
    preprocess_entries(data_dict)
    store_exp_data(data_dict, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data pre-processing')
    parser.add_argument('-i', '--input', metavar='input',
                        required=True, dest='input',
                        help='Data folder.')
    parser.add_argument('-o', '--output', metavar='output',
                        required=True, dest='output',
                        help='Output folder.')
    main(parser.parse_args())
