#!/bin/sh
# List every CAN interface with its USB bus-info and link state. Use the bus-info values to set
# CAN_LEFT_USB / CAN_RIGHT_USB for can_config.sh.
if ! dpkg -l | grep -q "can-utils"; then
    echo "Error: can-utils is not installed."
    echo "Install it with: sudo apt update && sudo apt install can-utils"
    exit 1
fi

can_interfaces=$(ip -br link show type can | awk '{print $1}')
if [ -z "$can_interfaces" ]; then
    echo "No CAN interface found."
    exit 0
fi

for iface in $can_interfaces; do
    BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
    if [ -z "$BUS_INFO" ]; then
        echo "Error: cannot read the bus-info of $iface."
        continue
    fi
    IS_LINK_UP=$(ip link show "$iface" | grep -q "UP" && echo "True" || echo "False")
    echo "Interface: $iface"
    echo "Bus-info : $BUS_INFO"
    echo "Link up  : $IS_LINK_UP"
done
