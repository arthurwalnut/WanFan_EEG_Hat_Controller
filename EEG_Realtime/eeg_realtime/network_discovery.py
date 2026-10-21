"""Read-only Windows IPv4 interfaces and existing ARP neighbours.

Uses IP Helper APIs, without CIM, administrator access, ping, TCP probes, or
subnet enumeration. A neighbour is only a discovery candidate, never identity.
"""
from dataclasses import dataclass
import ctypes
import ipaddress
import os
import socket


@dataclass(frozen=True)
class DiscoveryInterface:
    address: str
    prefix_length: int
    index: int
    neighbors: tuple[str, ...] = ()

    @property
    def network(self):
        return ipaddress.IPv4Network((self.address, self.prefix_length), strict=False)


def assemble_interfaces(adapters, neighbors):
    """Normalize API rows and restrict neighbours to their own local interface."""
    result, seen = [], set()
    local_addresses = {row[1] for row in adapters}
    for index, address_text, mask in adapters:
        try:
            address = ipaddress.IPv4Address(address_text)
            network = ipaddress.IPv4Network((address_text, mask), strict=False)
        except (ValueError, TypeError):
            continue
        if (address.is_unspecified or address.is_loopback or address.is_multicast
                or network.prefixlen == 0 or address_text in seen):
            continue
        candidates = set()
        for neighbor_index, peer_text, kind, mac in neighbors:
            if neighbor_index != index or kind not in (3, 4) or not mac or not any(mac):
                continue
            try:
                peer = ipaddress.IPv4Address(peer_text)
            except (ValueError, TypeError):
                continue
            if (peer in network and peer_text not in local_addresses and not peer.is_multicast
                    and not peer.is_unspecified and peer not in (network.network_address, network.broadcast_address)):
                candidates.add(str(peer))
        selected = tuple(sorted(candidates, key=ipaddress.IPv4Address)[:32])
        result.append(DiscoveryInterface(str(address), network.prefixlen, int(index), selected))
        seen.add(address_text)
    return result


def get_discovery_interfaces():
    if os.name != "nt":
        return []
    # Fixed-width fields matter: DWORD is 32 bits, pointers use host alignment.
    DWORD, BYTE = ctypes.c_uint32, ctypes.c_ubyte

    class AddressString(ctypes.Structure):
        pass
    AddressString._fields_ = [("Next", ctypes.POINTER(AddressString)),
        ("Address", ctypes.c_char * 16), ("Mask", ctypes.c_char * 16), ("Context", DWORD)]

    class AdapterHead(ctypes.Structure):
        pass
    # Only read the stable prefix through IpAddressList; do not model time_t
    # lease fields whose width differs between 32-bit SDK configurations.
    AdapterHead._fields_ = [("Next", ctypes.POINTER(AdapterHead)), ("ComboIndex", DWORD),
        ("AdapterName", ctypes.c_char * 260), ("Description", ctypes.c_char * 132),
        ("AddressLength", DWORD), ("Address", BYTE * 8), ("Index", DWORD),
        ("Type", DWORD), ("DhcpEnabled", DWORD), ("CurrentIpAddress", ctypes.POINTER(AddressString)),
        ("IpAddressList", AddressString)]

    class InterfaceRow(ctypes.Structure):
        _fields_ = [("Name", ctypes.c_uint16 * 256)] + [(name, DWORD) for name in
            ("Index", "Type", "Mtu", "Speed", "PhysAddrLen")] + [("PhysAddr", BYTE * 8)] + [
            (name, DWORD) for name in ("AdminStatus", "OperStatus", "LastChange", "InOctets",
            "InUcastPkts", "InNUcastPkts", "InDiscards", "InErrors", "InUnknownProtos",
            "OutOctets", "OutUcastPkts", "OutNUcastPkts", "OutDiscards", "OutErrors",
            "OutQLen", "DescrLen")] + [("Descr", BYTE * 256)]

    class NeighborRow(ctypes.Structure):
        _fields_ = [("Index", DWORD), ("PhysAddrLen", DWORD), ("PhysAddr", BYTE * 8),
                    ("Address", DWORD), ("Type", DWORD)]

    api = ctypes.WinDLL("iphlpapi.dll")
    api.GetAdaptersInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(DWORD)]
    api.GetAdaptersInfo.restype = DWORD
    api.GetIfEntry.argtypes = [ctypes.POINTER(InterfaceRow)]
    api.GetIfEntry.restype = DWORD
    api.GetIpNetTable.argtypes = [ctypes.c_void_p, ctypes.POINTER(DWORD), ctypes.c_int]
    api.GetIpNetTable.restype = DWORD

    def table(function, *extra):
        size = DWORD(15_000)
        for _ in range(3):
            if not 0 < size.value <= 2_000_000:
                raise OSError("Windows network table exceeds the bounded buffer")
            buffer = ctypes.create_string_buffer(size.value)
            code = int(function(buffer, ctypes.byref(size), *extra))
            if code == 0:
                return buffer
            if code in (111, 122):  # BUFFER_OVERFLOW / INSUFFICIENT_BUFFER
                continue
            if code == 232:  # NO_DATA
                return None
            raise OSError(code, "Cannot read Windows IPv4 network information")
        raise OSError("Windows network table changed repeatedly")

    adapter_buffer = table(api.GetAdaptersInfo)
    if adapter_buffer is None:
        return []
    adapters = []
    pointer = ctypes.cast(adapter_buffer, ctypes.POINTER(AdapterHead))
    for _ in range(64):
        if not pointer:
            break
        item = pointer.contents
        interface = InterfaceRow()
        interface.Index = item.Index
        # Keep Wi-Fi Direct virtual adapters: a hotspot need not have a gateway
        # or be a physical NIC. Only exclude loopback/tunnel and down interfaces.
        if item.Type not in (24, 131) and api.GetIfEntry(ctypes.byref(interface)) == 0 and interface.OperStatus == 5:
            address_pointer = ctypes.pointer(item.IpAddressList)
            for _ in range(64):
                if not address_pointer:
                    break
                row = address_pointer.contents
                adapters.append((int(item.Index), row.Address.decode("ascii"), row.Mask.decode("ascii")))
                address_pointer = row.Next
        pointer = item.Next
    neighbors = []
    try:
        neighbor_buffer = table(api.GetIpNetTable, False)
    except OSError:
        neighbor_buffer = None  # Broadcast discovery remains available.
    if neighbor_buffer is not None:
        count = DWORD.from_buffer(neighbor_buffer).value
        if count > (len(neighbor_buffer) - 4) // ctypes.sizeof(NeighborRow):
            raise OSError("Windows neighbor table size is inconsistent")
        rows = (NeighborRow * count).from_buffer(neighbor_buffer, 4)
        for row in rows:
            if 0 < row.PhysAddrLen <= 8:
                address = socket.inet_ntoa(int(row.Address).to_bytes(4, "little"))
                neighbors.append((int(row.Index), address, int(row.Type), bytes(row.PhysAddr[:row.PhysAddrLen])))
    return assemble_interfaces(adapters, neighbors)
