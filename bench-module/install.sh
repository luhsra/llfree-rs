qemu-system-x86_64 -m 28G -smp 4 \
  -enable-kvm \
  -hda hda.qcow2 \
  -boot d \
  -cdrom ~/Downloads/debian-11.2.0-amd64-netinst.iso \
# -curses \
# -nographic \
# -netdev user,id=mynet \
# -nic user,hostfwd=tcp:127.0.0.1:2222-:22

# https://wiki.debian.org/QEMU
# https://fosspost.org/tutorials/use-qemu-test-operating-systems-distributions
# https://wiki.qemu.org/Documentation/Networking
# Bootparameter
# vm 0000 (root 0000)
