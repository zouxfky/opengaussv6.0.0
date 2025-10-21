# vecexecutor 算子级 SIMD 优化分析报告

## 一、项目概述

`vecexecutor` 是 openGauss 的向量化执行引擎，包含完整的向量化算子集合。本报告按**算子（Operator）**维度分析 SIMD 优化潜力。

### 核心架构
- **数据结构**: VectorBatch（批量数据容器，默认1000行/批）
- **算子节点**: vecnode/ 目录包含所有向量化算子
- **表达式引擎**: vecexpression.cpp 负责表达式求值
- **基础运算**: vecprimitive/ 提供类型特化的向量运算

---

## 二、扫描算子（Scan Operators）

### 2.1 VecCStoreScan（列存扫描）

**文件**: `vecnode/veccstore.cpp`

**功能**: 列存储表的向量化扫描

**SIMD 优化潜力**: ⭐⭐⭐⭐⭐

**关键热点**:
1. **列数据解压缩**: CU（Compression Unit）解压
2. **谓词过滤**: WHERE 条件下推
3. **列数据拷贝**: 从存储层拷贝到 VectorBatch

**优化点**:
```cpp
// 典型场景: SELECT * FROM table WHERE col1 > 100
// 1. 批量解压缩（如果使用RLE/Delta编码）
// 2. SIMD 谓词过滤
for (i = 0; i < nrows; i++) {
    if (col_values[i] > threshold) {
        result[output_idx++] = col_values[i];
    }
}

// SIMD 优化方案:
__m256i data = _mm256_loadu_si256((__m256i*)&col_values[i]);
__m256i thresh = _mm256_set1_epi32(threshold);
__m256i mask = _mm256_cmpgt_epi32(data, thresh);
// 使用掩码进行 gather/compress 操作
```

**预期加速比**: 3-5x（谓词过滤场景）

**优先级**: P0 - 扫描是所有查询的基础操作

---

### 2.2 VecIndexScan（索引扫描）

**文件**: `vecnode/veccstoreindexscan.cpp`

**功能**: 基于索引的列存扫描

**SIMD 优化潜力**: ⭐⭐⭐

**关键操作**:
1. **Bitmap 位图操作**: 索引返回的行号位图
2. **稀疏数据收集**: 根据位图收集数据

**优化点**:
```cpp
// Bitmap 扫描：批量检查位图
// 当前: 逐位检查
for (i = 0; i < nrows; i++) {
    if (bitmap[i/8] & (1 << (i%8))) {
        fetch_row(i);
    }
}

// SIMD 优化: 并行处理8个字节的位图
__m64 bitmap_chunk = *(__m64*)&bitmap[i/8];
int mask = _mm_movemask_pi8(_mm_cmpgt_pi8(bitmap_chunk, zero));
// 根据 mask 批量获取数据
```

**预期加速比**: 2-3x

**优先级**: P1

---

## 三、连接算子（Join Operators）

### 3.1 VecHashJoin（哈希连接）

**文件**: `vecnode/vechashjoin.cpp`

**功能**: 向量化哈希连接，OLAP 查询中最常用的连接方式

**SIMD 优化潜力**: ⭐⭐⭐⭐⭐

**关键阶段**:

#### Phase 1: Build 阶段（构建哈希表）
```cpp
// 1.1 Hash 值计算 - 高度适合 SIMD
for (i = 0; i < batch->m_rows; i++) {
    hash_values[i] = hash_func(key_values[i]);
}

// SIMD 优化: 并行计算多个 hash
// 使用 CRC32 指令或自定义 SIMD hash
__m256i keys = _mm256_loadu_si256((__m256i*)&key_values[i]);
__m256i hashes = simd_hash_function(keys);
```

#### Phase 2: Probe 阶段（探测匹配）
```cpp
// 2.1 批量 hash 计算（同 Build）
// 2.2 批量哈希表查找
for (i = 0; i < batch->m_rows; i++) {
    hash = hash_values[i];
    bucket = hash % table_size;
    // 遍历冲突链
    for (entry = table[bucket]; entry; entry = entry->next) {
        if (entry->hash == hash && equals(entry->key, probe_key[i])) {
            output_match(entry, probe_row[i]);
        }
    }
}

// SIMD 优化: 批量比较
// 1. SIMD 计算多个 bucket 位置
// 2. 预取多个 bucket 的数据
// 3. SIMD 比较 key 值（对于整数/浮点 key）
```

**具体优化点**:

1. **整数 Key 的批量比较**
```cpp
// 假设 key 是 int64
__m256i probe_keys = _mm256_loadu_si256((__m256i*)&keys[i]);
__m256i table_keys = _mm256_loadu_si256((__m256i*)&hash_table_keys[bucket]);
__m256i cmp = _mm256_cmpeq_epi64(probe_keys, table_keys);
int match_mask = _mm256_movemask_epi8(cmp);
```

2. **NULL 值批量处理**
```cpp
// 批量检查 NULL 标记
__m256i null_flags = _mm256_loadu_si256((__m256i*)&flags[i]);
__m256i null_mask = _mm256_cmpeq_epi8(null_flags, zero);
```

3. **Prefetch 优化**
```cpp
// 预取下几个要访问的 bucket
for (int j = 0; j < PREFETCH_DISTANCE; j++) {
    _mm_prefetch(&table[hash_values[i+j] % table_size], _MM_HINT_T0);
}
```

**预期加速比**: 
- Hash 计算: 3-4x
- Key 比较（整数）: 4-6x
- 整体 Join: 1.5-2.5x

**优先级**: P0 - Hash Join 是 OLAP 最核心算子

---

### 3.2 VecMergeJoin（归并连接）

**文件**: `vecnode/vecmergejoin.cpp`

**功能**: 有序数据的归并连接

**SIMD 优化潜力**: ⭐⭐⭐

**关键操作**:
```cpp
// 双指针扫描，比较两个有序数组
while (left_idx < left_rows && right_idx < right_rows) {
    if (left_keys[left_idx] == right_keys[right_idx]) {
        output_match();
    } else if (left_keys[left_idx] < right_keys[right_idx]) {
        left_idx++;
    } else {
        right_idx++;
    }
}

// SIMD 优化: 批量比较
// 加载多个 key 进行并行比较
__m256i left = _mm256_loadu_si256((__m256i*)&left_keys[left_idx]);
__m256i right = _mm256_loadu_si256((__m256i*)&right_keys[right_idx]);
__m256i cmp_eq = _mm256_cmpeq_epi64(left, right);
__m256i cmp_lt = _mm256_cmpgt_epi64(right, left);
```

**预期加速比**: 2-3x

**优先级**: P1

---

### 3.3 VecNestLoop（嵌套循环连接）

**文件**: `vecnode/vecnestloop.cpp`

**功能**: 嵌套循环连接（小表场景）

**SIMD 优化潜力**: ⭐⭐⭐⭐

**关键操作**:
```cpp
// 外层表每一行与内层表所有行做笛卡尔积
for (outer_i = 0; outer_i < outer_rows; outer_i++) {
    for (inner_i = 0; inner_i < inner_rows; inner_i++) {
        if (join_condition(outer[outer_i], inner[inner_i])) {
            output_row();
        }
    }
}

// SIMD 优化: 广播外层值，批量与内层比较
__m256i outer_val = _mm256_set1_epi64x(outer_keys[outer_i]);
for (inner_i = 0; inner_i < inner_rows; inner_i += 4) {
    __m256i inner_vals = _mm256_loadu_si256((__m256i*)&inner_keys[inner_i]);
    __m256i match = _mm256_cmpeq_epi64(outer_val, inner_vals);
    // 处理匹配结果
}
```

**预期加速比**: 3-5x

**优先级**: P1

---

## 四、聚合算子（Aggregation Operators）

### 4.1 VecHashAgg（哈希聚合）

**文件**: `vecnode/vechashagg.cpp`, `vectorsonic/vsonichashagg.cpp`

**功能**: GROUP BY 哈希聚合

**SIMD 优化潜力**: ⭐⭐⭐⭐⭐

**关键操作**:

#### 4.1.1 分组 Key 的 Hash 计算
```cpp
// 同 HashJoin，批量计算 hash
for (i = 0; i < nrows; i++) {
    hash_values[i] = hash_func(group_keys[i]);
}

// SIMD: 并行 hash 计算（同 HashJoin）
```

#### 4.1.2 聚合函数计算

**SUM 聚合**:
```cpp
// 当前实现 (int4.inl:433-485)
for(i = 0; i < nrows; i++) {
    cell = loc[i];
    if(cell && IS_NULL(flag[i]) == false) {
        if(IS_NULL(cell->m_val[idx].flag)) {
            cell->m_val[idx].val = pVal[i];
            SET_NOTNULL(cell->m_val[idx].flag);
        } else {
            cell->m_val[idx].val += pVal[i];
        }
    }
}

// SIMD 优化: 向量化累加
// 对于同一个 group 的连续行，可以批量累加
__m256i acc = _mm256_setzero_si256();
for (int j = group_start; j < group_end; j += 4) {
    __m256i vals = _mm256_loadu_si256((__m256i*)&pVal[j]);
    acc = _mm256_add_epi64(acc, vals);
}
// 水平求和
cell->m_val[idx].val += horizontal_sum(acc);
```

**MIN/MAX 聚合**:
```cpp
// 当前实现 (int8.inl:89-129)
if (sop == SOP_GT)
    result = (args[0] > args[1]) ? args[0] : args[1];
else
    result = (args[0] < args[1]) ? args[0] : args[1];

// SIMD 优化
__m256i curr_max = _mm256_set1_epi64x(cell->m_val[idx].val);
for (int j = group_start; j < group_end; j += 4) {
    __m256i vals = _mm256_loadu_si256((__m256i*)&pVal[j]);
    curr_max = _mm256_max_epi64(curr_max, vals);
}
cell->m_val[idx].val = horizontal_max(curr_max);
```

**AVG 聚合**:
```cpp
// 需要维护 sum 和 count 两个值
// SIMD: 同时更新 sum（向量化加法）和 count（向量化计数）
```

**COUNT 聚合**:
```cpp
// 当前实现: 逐个计数
for (i = 0; i < nrows; i++) {
    if (NOT_NULL(flag[i])) count++;
}

// SIMD 优化: 批量 popcount
__m256i flags = _mm256_loadu_si256((__m256i*)&flag[i]);
__m256i not_null = _mm256_cmpeq_epi8(flags, _mm256_set1_epi8(1));
int mask = _mm256_movemask_epi8(not_null);
count += _mm_popcnt_u32(mask);
```

**预期加速比**:
- SUM/COUNT: 4-6x
- MIN/MAX: 3-5x
- AVG: 3-5x
- 整体聚合查询: 2-3x

**优先级**: P0 - 聚合是 OLAP 核心操作

---

### 4.2 VecSortAgg（排序聚合）

**文件**: `vecnode/vecsortagg.cpp`

**功能**: 基于排序的聚合（数据已排序场景）

**SIMD 优化潜力**: ⭐⭐⭐⭐

**关键操作**:
```cpp
// 检测分组边界
for (i = 1; i < nrows; i++) {
    if (group_keys[i] != group_keys[i-1]) {
        // 输出前一个组的聚合结果
        output_group();
        // 开始新组
    }
    // 累加当前行到当前组
}

// SIMD 优化: 批量比较检测边界
__m256i prev = _mm256_loadu_si256((__m256i*)&keys[i-4]);
__m256i curr = _mm256_loadu_si256((__m256i*)&keys[i]);
__m256i neq = _mm256_xor_si256(
    _mm256_cmpeq_epi64(prev, curr),
    _mm256_set1_epi64x(-1)
);
int boundary_mask = _mm256_movemask_epi8(neq);
```

**预期加速比**: 2-4x

**优先级**: P1

---

## 五、排序算子（Sort Operator）

### 5.1 VecSort（向量排序）

**文件**: `vecnode/vecsort.cpp`

**功能**: 批量数据排序（ORDER BY）

**SIMD 优化潜力**: ⭐⭐⭐

**关键操作**:

#### 5.1.1 比较操作
```cpp
// 排序的核心是大量的比较操作
int compare(ScalarValue a, ScalarValue b) {
    return (a < b) ? -1 : ((a > b) ? 1 : 0);
}

// SIMD 优化: 批量比较
// 用于基数排序或分区操作
__m256i data = _mm256_loadu_si256((__m256i*)&values[i]);
__m256i pivot = _mm256_set1_epi64x(pivot_value);
__m256i cmp = _mm256_cmpgt_epi64(data, pivot);
int mask = _mm256_movemask_epi8(cmp);
// 根据 mask 进行分区
```

#### 5.1.2 数据移动
```cpp
// 排序过程中的数据交换和移动
// SIMD 优化: 向量化的数据搬移
```

**已有优化**: 代码中可能已使用 qsort 等优化算法

**SIMD 收益**: 有限（排序本身算法复杂度占主导）

**预期加速比**: 1.5-2x

**优先级**: P2

---

## 六、其他算子

### 6.1 VecLimit（限制算子）

**文件**: `vecnode/veclimit.cpp`

**功能**: LIMIT / OFFSET

**SIMD 优化潜力**: ⭐

**原因**: 逻辑简单，主要是计数，SIMD收益很小

**优先级**: P3（不建议优化）

---

### 6.2 VecUnique（去重算子）

**文件**: `vecnode/vecunique.cpp`

**功能**: DISTINCT 去重

**SIMD 优化潜力**: ⭐⭐⭐

**关键操作**: 与 SortAgg 类似，检测连续重复

**优化方案**: 同 VecSortAgg

**预期加速比**: 2-3x

**优先级**: P1

---

### 6.3 VecMaterial（物化算子）

**文件**: `vecnode/vecmaterial.cpp`

**功能**: 物化中间结果

**SIMD 优化潜力**: ⭐⭐

**主要操作**: 批量数据拷贝

**优化**: 使用 _mm_stream_* 进行非临时性写入

**预期加速比**: 1.5-2x

**优先级**: P2

---

### 6.4 VecWindowAgg（窗口函数）

**文件**: `vecnode/vecwindowagg.cpp`

**功能**: 窗口函数（ROW_NUMBER, RANK, LAG/LEAD等）

**SIMD 优化潜力**: ⭐⭐⭐⭐

**关键操作**:

1. **ROW_NUMBER**: 连续编号
```cpp
// 简单递增，SIMD可批量生成序列号
__m256i base = _mm256_set1_epi64x(row_num);
__m256i offsets = _mm256_setr_epi64x(0, 1, 2, 3);
__m256i row_nums = _mm256_add_epi64(base, offsets);
```

2. **RANK/DENSE_RANK**: 需要比较分组边界
```cpp
// 批量比较，检测边界（同 SortAgg）
```

3. **窗口聚合（SUM/AVG OVER）**: 滑动窗口
```cpp
// 滑动窗口求和可用 SIMD 优化
__m256i sum = _mm256_setzero_si256();
for (int i = window_start; i < window_end; i += 4) {
    __m256i vals = _mm256_loadu_si256((__m256i*)&values[i]);
    sum = _mm256_add_epi64(sum, vals);
}
```

**预期加速比**: 2-4x

**优先级**: P1

---

## 七、表达式计算（跨算子通用）

### 7.1 VecExpression（表达式引擎）

**文件**: `vecexpression.cpp`

**功能**: 所有算子共用的表达式求值引擎

**SIMD 优化潜力**: ⭐⭐⭐⭐⭐

**关键函数**:

#### 7.1.1 算术表达式
```cpp
// a + b * c - d
// 当前: 逐行计算
for (i = 0; i < nrows; i++) {
    result[i] = a[i] + b[i] * c[i] - d[i];
}

// SIMD 优化: 批量计算
__m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
__m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
__m256i vc = _mm256_loadu_si256((__m256i*)&c[i]);
__m256i vd = _mm256_loadu_si256((__m256i*)&d[i]);
__m256i tmp = _mm256_mul_epi64(vb, vc);
tmp = _mm256_add_epi64(va, tmp);
__m256i vresult = _mm256_sub_epi64(tmp, vd);
```

#### 7.1.2 布尔表达式
```cpp
// WHERE col1 > 100 AND col2 < 200
__m256i col1 = _mm256_loadu_si256((__m256i*)&col1_vals[i]);
__m256i col2 = _mm256_loadu_si256((__m256i*)&col2_vals[i]);
__m256i cmp1 = _mm256_cmpgt_epi32(col1, _mm256_set1_epi32(100));
__m256i cmp2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(200), col2);
__m256i result = _mm256_and_si256(cmp1, cmp2);
```

**预期加速比**: 4-8x

**优先级**: P0 - 影响所有算子

---

## 八、优化优先级总结

### P0 级（必须优化，影响最大）

| 算子 | 关键操作 | 预期加速 | 影响范围 |
|------|---------|---------|---------|
| **VecExpression** | 算术/布尔表达式 | 4-8x | 所有算子 |
| **VecHashJoin** | Hash计算、Key比较 | 1.5-2.5x | 大部分JOIN |
| **VecHashAgg** | 聚合函数计算 | 2-3x | 所有聚合查询 |
| **VecCStoreScan** | 谓词过滤 | 3-5x | 所有扫描 |

### P1 级（重要优化，收益明显）

| 算子 | 关键操作 | 预期加速 | 影响范围 |
|------|---------|---------|---------|
| **VecSortAgg** | 边界检测、聚合 | 2-4x | 有序聚合 |
| **VecWindowAgg** | 窗口函数 | 2-4x | 分析查询 |
| **VecMergeJoin** | 批量比较 | 2-3x | 有序JOIN |
| **VecNestLoop** | 笛卡尔积过滤 | 3-5x | 小表JOIN |
| **VecUnique** | 去重检测 | 2-3x | DISTINCT |

### P2 级（次要优化，收益有限）

| 算子 | 关键操作 | 预期加速 | 影响范围 |
|------|---------|---------|---------|
| **VecSort** | 比较操作 | 1.5-2x | ORDER BY |
| **VecMaterial** | 数据拷贝 | 1.5-2x | 物化场景 |
| **VecIndexScan** | Bitmap操作 | 2-3x | 索引扫描 |

### P3 级（不建议优化）

- VecLimit: 逻辑过于简单
- VecResult: 无计算逻辑

---

## 九、实施建议

### 9.1 分阶段实施

**Phase 1（1个月）**: 表达式引擎 + 基础算子
- VecExpression 表达式 SIMD 化
- VecCStoreScan 谓词过滤优化
- 基础类型运算库（vecprimitive/）

**Phase 2（1个月）**: 核心 JOIN 和 AGG
- VecHashJoin 优化
- VecHashAgg 优化
- Hash 函数向量化

**Phase 3（1个月）**: 其他算子
- VecSortAgg, VecWindowAgg
- VecMergeJoin, VecNestLoop
- 测试与调优

### 9.2 技术架构

```cpp
// 1. SIMD 抽象层
namespace simd {
    // 运行时 CPU 特性检测
    enum class ISA { SSE42, AVX2, AVX512 };
    ISA detect_isa();
    
    // 函数分发
    template<typename T>
    void add_array(T* a, T* b, T* result, int n) {
        if (has_avx512()) {
            add_array_avx512(a, b, result, n);
        } else if (has_avx2()) {
            add_array_avx2(a, b, result, n);
        } else {
            add_array_scalar(a, b, result, n);
        }
    }
}

// 2. 算子级别集成
class VecHashJoin {
    void BuildHashTable() {
        // 使用 SIMD hash 函数
        simd::hash_batch(keys, hashes, nrows);
        ...
    }
};
```

### 9.3 benchmark 体系

```sql
-- 典型查询场景
-- Q1: 扫描 + 聚合
SELECT sum(quantity), avg(price) 
FROM lineitem 
WHERE shipdate < '1998-01-01';

-- Q2: JOIN + 聚合  
SELECT o_orderkey, sum(l_quantity)
FROM orders JOIN lineitem ON o_orderkey = l_orderkey
GROUP BY o_orderkey;

-- Q3: 窗口函数
SELECT order_id, amount, 
       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date)
FROM orders;
```

---

## 十、总结

**核心结论**: vecexecutor 的主要算子**高度适合** SIMD 优化

**最有价值的优化目标**:
1. ✅ **表达式引擎**（影响所有算子）
2. ✅ **HashJoin/HashAgg**（OLAP 核心）
3. ✅ **Scan + 谓词过滤**（查询入口）

**预期整体收益**:
- **TPC-H Q1**: 2-3x 加速
- **TPC-H Q6**: 3-4x 加速（大量过滤）
- **典型聚合查询**: 2-2.5x 加速
- **复杂 JOIN 查询**: 1.5-2x 加速

**投入产出比**: ⭐⭐⭐⭐⭐

预计 **3-4 人月**投入，可获得 **50-150%** 的 OLAP 查询性能提升。

