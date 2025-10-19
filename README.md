# openGauss 6.0.0 RISC-V 编译安装指南

> **本仓库已完成所有前置步骤（解压、打补丁），克隆后可直接编译！**

## ⚡ 快速开始（推荐）

**如果你从本 GitHub 仓库克隆，所有补丁已应用，直接编译即可：**

```bash
# 1. 克隆本仓库
git clone https://github.com/zouxfky/opengaussv6.0.0.git
cd opengaussv6.0.0

# 2. 安装依赖（仅需一次）
sudo dnf install -y automake bison boost-devel cjson-devel cmake flex \
  gcc gcc-c++ git glibc-devel krb5-devel libcurl-devel libaio-devel \
  libxml2-devel libyaml-devel ncurses-devel openldap-devel openssl-devel \
  pam-devel patch perl-ExtUtils-Embed python3-devel readline-devel zlib-devel

# 3. 直接编译
mkdir build && cd build
export DEBUG_TYPE=release ENABLE_LITE_MODE=ON
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/opengauss \
  -DENABLE_MULTIPLE_NODES=OFF \
  -DENABLE_LITE_MODE=ON \
  -DENABLE_OPENEULER_MAJOR=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_OPENEULER_OS=ON

make -j$(nproc)
sudo make install
```

**完成！🎉** 现在可以初始化和启动数据库了（见下方"测试运行"章节）。

---

## 📌 两种源码获取方式对比

| 源码来源 | 前置步骤 | 编译难度 | 适用场景 |
|---------|---------|---------|----------|
| **本 GitHub 仓库**<br>`github.com/zouxfky/opengaussv6.0.0` | ✅ 已完成<br>（已解压、已打补丁） | ⭐ 简单<br>克隆后直接编译 | 🚀 **快速开发、测试、学习**<br>（推荐） |
| **Gitee 原始仓库**<br>`gitee.com/opengauss/riscv` | ❌ 需手动完成<br>（解压、打补丁） | ⭐⭐⭐ 复杂<br>需执行 5 大步骤 | 📚 深入学习补丁机制<br>自定义修改 |

> 💡 **说明**：
> - **本仓库**（GitHub）：所有第三方库已解压，所有 RISC-V 补丁已应用，`xgboost` Git 问题已修复
> - **原始仓库**（Gitee）：仅包含压缩包和补丁文件，需要手动执行解压和打补丁步骤（见下方"从 Gitee 手动构建"章节）

## 📋 前提条件

### 系统要求
- **操作系统**：openEuler 24.03 LTS RISC-V 或更高版本
- **架构**：RISC-V 64位
- **内存**：建议 ≥ 8GB（编译时需要）
- **磁盘空间**：≥ 10GB

### 依赖包安装

```bash
sudo dnf install -y \
  automake bison boost-devel cjson-devel cmake flex \
  gcc gcc-c++ git glibc-devel krb5-devel libcurl-devel \
  libaio-devel libxml2-devel libyaml-devel ncurses-devel \
  openldap-devel openssl-devel pam-devel patch \
  perl-ExtUtils-Embed python3-devel readline-devel zlib-devel
```

---

## 🔧 从 Gitee 手动构建（可选）

> ⚠️ **注意**：如果你从**本 GitHub 仓库**克隆，**跳过本章节**！直接看顶部"快速开始"。
> 
> 以下步骤仅适用于从 **Gitee 原始仓库** (`gitee.com/opengauss/riscv`) 克隆的情况。

### 完整步骤摘要（Gitee 仓库）

```bash
# 1. 克隆并解压（仅 Gitee 用户需要）
git clone https://gitee.com/opengauss/riscv.git && cd riscv && git checkout v6.0.0
tar -xzf openGauss-server-v6.0.0.tar.gz && cd openGauss-server-v6.0.0

# 2. 解压第三方库
mkdir -p 3rd/{DCF,aws-sdk-cpp,xgboost}
tar -xzf ../DCF-5.1.0.tar.gz --strip-components=1 -C 3rd/DCF
tar -xzf ../aws-sdk-cpp-1.11.327.tar.gz --strip-components=1 -C 3rd/aws-sdk-cpp
tar -xzf ../xgboost-v1.4.1.tar.gz --strip-components=1 -C 3rd/xgboost
tar -xzf ../dmlc-core-v0.5.tar.gz --strip-components=1 -C 3rd/xgboost/dmlc-core

# 3. 应用补丁
for p in using-system-package-instead-binarylibs add-riscv64-support \
  Fix-pointer-comparison-syntax-error link-gaussdb-with-atomic \
  integrate-3rd-source-code; do patch -p1 < ../${p}.patch; done
patch -p1 -d 3rd/DCF < ../add-riscv64-support-on-DCF.patch
patch -p1 -d 3rd/xgboost < ../add-compile-options-to-xgboost.patch

# 4. 修复 cJSON (Linux 版本)
sed -i 's|#include "external/cJSON.h"|#include <cjson/cJSON.h>|g' \
  3rd/aws-sdk-cpp/crt/aws-crt-cpp/crt/aws-c-common/source/json.c

# 5. 编译
mkdir build && cd build
export DEBUG_TYPE=release ENABLE_LITE_MODE=ON
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/opengauss -DENABLE_MULTIPLE_NODES=OFF \
  -DENABLE_LITE_MODE=ON -DENABLE_OPENEULER_MAJOR=ON -DCMAKE_BUILD_TYPE=Release \
  -DWITH_OPENEULER_OS=ON
make -j$(nproc) && sudo make install
```

---

### 详细步骤说明（Gitee 仓库）

#### 1. 克隆 Gitee 仓库

```bash
git clone https://gitee.com/opengauss/riscv.git
cd riscv
git checkout v6.0.0
```

#### 2. 解压源码

```bash
# 解压主源码
tar -xzf openGauss-server-v6.0.0.tar.gz
cd openGauss-server-v6.0.0
```

#### 3. 解压第三方依赖

```bash
# DCF（分布式一致性框架）
mkdir -p 3rd/DCF
tar -xzf ../DCF-5.1.0.tar.gz --strip-components=1 -C 3rd/DCF

# AWS SDK C++
mkdir -p 3rd/aws-sdk-cpp
tar -xzf ../aws-sdk-cpp-1.11.327.tar.gz --strip-components=1 -C 3rd/aws-sdk-cpp

# XGBoost（机器学习库）
mkdir -p 3rd/xgboost
tar -xzf ../xgboost-v1.4.1.tar.gz --strip-components=1 -C 3rd/xgboost

# DMLC-Core（XGBoost 依赖）
tar -xzf ../dmlc-core-v0.5.tar.gz --strip-components=1 -C 3rd/xgboost/dmlc-core
```

#### 4. 应用补丁（必须）

> ⚠️ **重要**：必须手动应用所有补丁，否则编译会失败（尤其是 RISC-V 架构支持和原子操作链接）。

```bash
# 应用主源码补丁
patch -p1 < ../using-system-package-instead-binarylibs.patch
patch -p1 < ../add-riscv64-support.patch
patch -p1 < ../Fix-pointer-comparison-syntax-error.patch
patch -p1 < ../link-gaussdb-with-atomic.patch
patch -p1 < ../integrate-3rd-source-code.patch

# 应用第三方库补丁（注意 -d 参数指定目录）
patch -p1 -d 3rd/DCF < ../add-riscv64-support-on-DCF.patch
patch -p1 -d 3rd/xgboost < ../add-compile-options-to-xgboost.patch

# 可选：SM2 密钥对支持
# patch -p1 < ../Using-sm2-curve-generate-key-pair.patch
```

**验证补丁应用成功**：

```bash
# 检查关键修改是否生效
grep -r "latomic" CMakeLists.txt src/bin/*/CMakeLists.txt | head -3
```

#### 5. 修复 cJSON 头文件路径（必须）

AWS SDK 需要 `cJSON.h`，但默认路径不对。**直接修改源文件**（推荐）：

**Linux (openEuler/Fedora)：**
```bash
sed -i 's|#include "external/cJSON.h"|#include <cjson/cJSON.h>|g' \
  3rd/aws-sdk-cpp/crt/aws-crt-cpp/crt/aws-c-common/source/json.c

# 验证修改
grep 'cJSON.h' 3rd/aws-sdk-cpp/crt/aws-crt-cpp/crt/aws-c-common/source/json.c
```

**macOS（仅用于本地测试，不推荐）：**
```bash
# macOS 的 sed 语法不同，-i 后面需要加 ''
sed -i '' 's|#include "external/cJSON.h"|#include <cjson/cJSON.h>|g' \
  3rd/aws-sdk-cpp/crt/aws-crt-cpp/crt/aws-c-common/source/json.c
```

> ⚠️ **注意**：应该在 RISC-V 服务器上编译，不要在本地 Mac 上编译！

**备选方案（符号链接）：**

```bash
mkdir -p 3rd/aws-sdk-cpp/crt/aws-crt-cpp/crt/aws-c-common/external
ln -sf /usr/include/cjson/cJSON.h \
  3rd/aws-sdk-cpp/crt/aws-crt-cpp/crt/aws-c-common/external/cJSON.h
```

#### 6. 配置构建

```bash
mkdir build
cd build

# 设置必要的环境变量（必须！）
export DEBUG_TYPE=release
export ENABLE_LITE_MODE=ON

# 配置 CMake（用 CMAKE_INSTALL_PREFIX 指定安装路径，避免路径拼接问题）
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/opengauss \
  -DENABLE_MULTIPLE_NODES=OFF \
  -DENABLE_PRIVATEGAUSS=OFF \
  -DENABLE_THREAD_SAFETY=ON \
  -DENABLE_LITE_MODE=ON \
  -DENABLE_OPENEULER_MAJOR=ON \
  -DENABLE_BBOX=OFF \
  -DTEST=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_OPENEULER_OS=ON

# 重要：不要设置 PREFIX_HOME 环境变量，会导致路径拼接错误！
# 直接用 CMAKE_INSTALL_PREFIX 指定安装路径
```

**环境变量说明：**
- `DEBUG_TYPE=release`：编译类型（release/debug）
- `ENABLE_LITE_MODE=ON`：轻量模式开关
- ❌ ~~`PREFIX_HOME`~~：不要设置！会导致路径拼接错误

**CMake 选项说明：**
- `CMAKE_INSTALL_PREFIX=/opt/opengauss`：安装目录（**必须用这个选项**，不要用 PREFIX_HOME）
- `ENABLE_MULTIPLE_NODES=OFF`：单节点模式（RISC-V 推荐）
- `ENABLE_LITE_MODE=ON`：轻量模式
- `ENABLE_OPENEULER_MAJOR=ON`：openEuler 优化
- `CMAKE_BUILD_TYPE=Release`：发布版本（优化编译）

#### 7. 编译

```bash
# 根据 CPU 核心数选择并行度
make -j$(nproc)

# 如果内存不足（< 4GB），降低并行度
# make -j2
# 或单线程编译
# make -j1
```

**预计编译时间：**
- 4 核心：约 60-90 分钟
- 单核心：约 3-4 小时

#### 8. 安装

```bash
# 安装到默认目录 /usr/local（需要 root 权限）
sudo make install

# 或在 cmake 配置时指定安装目录（推荐用于开发）
# cd ../
# rm -rf build && mkdir build && cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=/opt/opengauss [其他选项...]
# make -j$(nproc)
# sudo make install
```

**安装后配置**：

```bash
# 添加到 PATH（如果安装到非标准目录）
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
which gs_initdb
gs_initdb --version
```

---

## 🧪 测试运行

### 初始化数据库

```bash
# 创建数据库用户（不要用 root）
sudo useradd -m -s /bin/bash omm
sudo passwd omm

# 切换到数据库用户
su - omm

# 初始化数据库
/usr/local/bin/gs_initdb -D /home/omm/data --nodename=single_node -w "YourPassword@123"
```

### 启动数据库

```bash
# 启动服务
/usr/local/bin/gs_ctl start -D /home/omm/data

# 连接数据库
/usr/local/bin/gsql -d postgres -p 5432
```

### 停止数据库

```bash
/usr/local/bin/gs_ctl stop -D /home/omm/data
```

---

## ⚠️ 常见问题

### 1. CMake 配置错误

**问题**：`CMake Error: Could not find CMAKE_ROOT`

**解决**：清理旧的 CMake 缓存

```bash
rm -rf build
mkdir build
cd build
cmake ..
```

### 2. 路径拼接错误（重要！）

**问题**：CMake 配置时出现路径重复拼接
```
-- /path/to/project/path/to/project/build/path/to/project/build/install
-- Configuring incomplete, errors occurred!
```

**原因**：使用 `PREFIX_HOME` 环境变量时，CMakeLists.txt 会导致路径错误拼接

**解决**：不要使用 `PREFIX_HOME`，改用 `CMAKE_INSTALL_PREFIX`

```bash
# ❌ 错误方式
export PREFIX_HOME=/opt/opengauss
cmake .. [其他选项...]

# ✅ 正确方式
export DEBUG_TYPE=release ENABLE_LITE_MODE=ON
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/opengauss [其他选项...]
```

#### 3. 编译时找不到 cJSON.h

**问题**：`fatal error: external/cJSON.h: No such file or directory`

**解决**：确认 `cjson-devel` 已安装，并创建符号链接或修改源文件（见步骤 5）

#### 4. 内存不足导致编译失败

**问题**：`c++: internal compiler error: Killed (program cc1plus)`

**解决**：
- 降低编译并行度：`make -j1`
- 增加 swap 空间

#### 5. 链接错误（找不到 libatomic）

**问题**：`undefined reference to __atomic_*`

**解决**：这个问题应该已经通过 `link-gaussdb-with-atomic.patch` 解决

#### 6. 编译时 ereport 检查失败

**问题**：`ereport scan detect unstandarded message`

**解决**：在 CMake 配置时添加 `-DENABLE_EREPORT_VERIFICATION=OFF`

#### 7. dmlc-core 找不到源文件

**问题**：`Cannot find source file: src/io/indexed_recordio_split.cc`

**原因**：dmlc-core 的 `.gitignore` 可能忽略了某些文件

**解决**：确认 dmlc-core 解压完整

```bash
# 重新解压 dmlc-core
rm -rf 3rd/xgboost/dmlc-core/*
tar -xzf ../dmlc-core-v0.5.tar.gz --strip-components=1 -C 3rd/xgboost/dmlc-core

# 验证文件存在
ls -la 3rd/xgboost/dmlc-core/src/io/indexed_recordio_split.*
```

---

## 📂 目录结构

```
riscv/
├── openGauss-server-v6.0.0.tar.gz    # 主源码压缩包
├── DCF-5.1.0.tar.gz                   # DCF 分布式一致性框架
├── aws-sdk-cpp-1.11.327.tar.gz        # AWS SDK
├── xgboost-v1.4.1.tar.gz              # XGBoost 机器学习库
├── dmlc-core-v0.5.tar.gz              # XGBoost 依赖
├── *.patch                            # 各种补丁文件
└── openGauss-server-v6.0.0/           # 解压后的源码目录
    ├── 3rd/                           # 第三方库目录
    │   ├── DCF/
    │   ├── aws-sdk-cpp/
    │   └── xgboost/
    └── build/                         # 编译目录
```

---

## 📌 开发工作流

如果你想修改源码并管理版本，建议将准备好的源码上传到自己的 Git 仓库：

```bash
# 在 openGauss-server-v6.0.0 目录下
git init
git add .
git commit -m "Initial commit: openGauss 6.0.0 RISC-V with patches"

# 推送到你的远程仓库
git remote add origin https://gitee.com/your-username/opengauss-riscv.git
git push -u origin main
```

之后在其他机器上可以直接克隆你的仓库，跳过解压和打补丁步骤。

---

## 📚 参考资源

- **openGauss 官网**：https://opengauss.org/
- **Gitee 仓库**：https://gitee.com/opengauss/riscv
- **文档中心**：https://docs.opengauss.org/

---

## 🐛 问题反馈

如果遇到问题，请检查：
1. 依赖包是否全部安装
2. CMake 版本 ≥ 3.12
3. GCC 版本 ≥ 7.3
4. 系统内存是否充足

**技巧**：使用 `make -j1 VERBOSE=1` 查看详细的编译错误信息。

