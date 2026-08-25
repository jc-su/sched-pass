obj-m += nta_nvme_p2p.o

NVIDIA_P2P_INCLUDE ?= /usr/src/nvidia-$(shell cat /sys/module/nvidia/version)/nvidia-peermem
ccflags-y += -I$(src)/../../include -I$(NVIDIA_P2P_INCLUDE)
