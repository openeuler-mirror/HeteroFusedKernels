#!/bin/bash
set -e  # 遇到错误立即退出

# ==============================================
# 前置步骤：清理残留的共享内存和挂载点（核心新增部分）
# ==============================================
echo "===== 开始清理残留共享内存环境 ====="

# 1. 清理 System V 共享内存（未被使用的段）
echo "清理未使用的 System V 共享内存..."
if ipcs -m &> /dev/null; then
    # 只删除附着数为0的共享内存段（避免影响正在运行的进程）
    ipcs -m | awk '$6 == 0 {print "sudo ipcrm -m", $2}' | sh || true
fi

# 2. 清理 /dev/shm 中的残留文件（保留目录结构，删除文件）
echo "清理 /dev/shm 残留文件..."
if [ -d "/dev/shm" ]; then
    # 只删除文件，不删除子目录（避免误删其他程序的目录）
    sudo find /dev/shm -type f -delete
    # 可选：删除空目录（如果确认安全）
    # sudo find /dev/shm -type d -empty -delete
fi

# 3. 确保 /hugepages 挂载点完全卸载
echo "卸载可能残留的 /hugepages 挂载..."
while mount | grep -q " /hugepages"; do
    if ! umount "/hugepages"; then
        echo "警告：无法卸载 /hugepages，可能被进程占用"
        # 查找占用进程并提示
        echo "占用 /hugepages 的进程："
        fuser -v /hugepages || true
        exit 1
    fi
done

# 4. 清理 /hugepages 目录（如果存在）
if [ -d "/hugepages" ]; then
    echo "清理 /hugepages 目录内容..."
    sudo rm -rf /hugepages/*
fi

echo "===== 残留环境清理完成 ====="
echo

# ==============================================
# 原有配置步骤（保持不变，确保基于干净环境重建）
# ==============================================

# 1. 配置大页数量（如果尚未设置）
target_hugepages=2048
current_hugepages=$(sysctl -n vm.nr_hugepages)
if [ "$current_hugepages" -ne "$target_hugepages" ]; then
    echo "设置大页数量为 $target_hugepages..."
    sudo sysctl -w vm.nr_hugepages=$target_hugepages
    sudo sysctl -p
else
    echo "大页数量已为 $target_hugepages，无需修改"
fi

# 2. 重新配置 /dev/shm（确保基于干净状态挂载）
shm_mount_point="/dev/shm"
desired_shm_opts="size=4194304k,huge=always"

if mount | grep -q "$shm_mount_point type tmpfs" && \
   mount | grep -q "$shm_mount_point.*$desired_shm_opts"; then
    echo "/dev/shm 已按预期挂载，无需操作"
else
    echo "重新配置 /dev/shm..."
    if mount | grep -q "$shm_mount_point"; then
        sudo umount "$shm_mount_point"
    fi
    sudo mount -t tmpfs -o "$desired_shm_opts" tmpfs "$shm_mount_point"
fi

# 3. 配置 /hugepages 大页文件系统
huge_mount_point="/hugepages"
desired_huge_opts="pagesize=2M"

if mount | grep -q " $huge_mount_point type hugetlbfs" && \
   mount | grep -q " $huge_mount_point.*$desired_huge_opts"; then
    echo "/hugepages 已按预期挂载，无需操作"
else
    echo "配置 /hugepages 大页文件系统..."
    sudo mkdir -p "$huge_mount_point"
    sudo mount -t hugetlbfs none "$huge_mount_point" -o "$desired_huge_opts"
fi

# 4. 导入环境变量（避免重复导入）
if [ -f "$env_script" ]; then
    if [ -z "$ASCEND_TOOLKIT_SET" ]; then
        echo "导入环境变量: $env_script"
        source "$env_script"
        export ASCEND_TOOLKIT_SET=1
    else
        echo "环境变量已导入，跳过"
    fi
else
    echo "警告: 环境变量脚本 $env_script 不存在"
fi

echo "所有配置完成，环境已重置并准备就绪"

