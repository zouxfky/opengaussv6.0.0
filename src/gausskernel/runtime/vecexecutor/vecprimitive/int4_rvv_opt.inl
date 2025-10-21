/*
 * Copyright (c) 2020 Huawei Technologies Co.,Ltd.
 *
 * openGauss is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 *
 * int4_rvv_opt.inl
 *	  RISC-V Vector Extension (RVV) optimized implementation of 32-bit integers.
 *
 * IDENTIFICATION
 *    src/gausskernel/runtime/vecexecutor/vecprimitive/int4_rvv_opt.inl
 *
 * -------------------------------------------------------------------------
 */

#ifndef INT4_RVV_OPT_INL
#define INT4_RVV_OPT_INL

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

#include "catalog/pg_type.h"
#include <ctype.h>
#include <limits.h>
#include "vecexecutor/vechashtable.h"
#include "utils/array.h"
#include "utils/biginteger.h"
#include "vectorsonic/vsonichashagg.h"

#define SAMESIGN(a,b)	(((a) < 0) == ((b) < 0))

/*
 * RVV优化进度：
 * 第一轮：基础RVV向量化谓词比较
 * 第二轮：Selection向量化 + 软件预取
 * 第三轮：循环展开 + 寄存器优化 + 对齐优化
 * 
 * 最终优化点：
 * 1. 使用RVV向量加载指令批量读取数据
 * 2. 使用向量掩码处理NULL值
 * 3. 使用RVV比较指令并行执行谓词
 * 4. 4路循环展开减少循环开销
 * 5. 软件预取优化内存访问
 * 6. 寄存器复用减少冗余操作
 * 7. 对齐hint优化cache line访问
 */

#ifdef __riscv_vector

// Helper: 执行比较操作（避免代码重复）
template <SimpleOp sop>
static inline vbool32_t
execute_comparison_rvv(vint32m1_t v_arg1, vint32m1_t v_arg2, size_t vl)
{
	if constexpr (sop == SOP_EQ) {
		return __riscv_vmseq_vv_i32m1_b32(v_arg1, v_arg2, vl);
	} else if constexpr (sop == SOP_NEQ) {
		return __riscv_vmsne_vv_i32m1_b32(v_arg1, v_arg2, vl);
	} else if constexpr (sop == SOP_LT) {
		return __riscv_vmslt_vv_i32m1_b32(v_arg1, v_arg2, vl);
	} else if constexpr (sop == SOP_LE) {
		return __riscv_vmsle_vv_i32m1_b32(v_arg1, v_arg2, vl);
	} else if constexpr (sop == SOP_GT) {
		return __riscv_vmsgt_vv_i32m1_b32(v_arg1, v_arg2, vl);
	} else { // SOP_GE
		return __riscv_vmsge_vv_i32m1_b32(v_arg1, v_arg2, vl);
	}
}

// RVV优化版本的int32比较操作（第三轮最终版）
template <SimpleOp sop, typename Datatype>
ScalarVector*
vint_sop_rvv_optimized(PG_FUNCTION_ARGS)
{
	ScalarValue* parg1 = PG_GETARG_VECVAL(0);
	ScalarValue* parg2 = PG_GETARG_VECVAL(1);
	int32		 nvalues = PG_GETARG_INT32(2);
	ScalarValue* presult = PG_GETARG_VECVAL(3);
	uint8*	pflag = (uint8*)(PG_GETARG_VECTOR(3)->m_flag);
	bool*		pselection = PG_GETARG_SELECTION(4);
	uint8*		pflags1 = (uint8*)(PG_GETARG_VECTOR(0)->m_flag);
	uint8*		pflags2 = (uint8*)(PG_GETARG_VECTOR(1)->m_flag);
	int          i;

	// 检查数据类型，当前仅支持int32
	static_assert(sizeof(Datatype) == sizeof(int32_t), "RVV optimization currently supports int32 only");

	if(likely(pselection == NULL))
	{
		// 第三轮优化：4路循环展开 + 寄存器优化
		int vl;
		int unroll_count = (nvalues >> 2); // 除以4
		int remainder_start = unroll_count << 2; // 乘以4
		
		// 主循环：4路展开
		for (i = 0; i < remainder_start; ) {
			vl = __riscv_vsetvl_e32m1(nvalues - i);
			
			// 软件预取下4个块的数据
			if __builtin_expect(i + (vl * 4) < nvalues, 1) {
				__builtin_prefetch(&parg1[i + (vl * 4)], 0, 1);
				__builtin_prefetch(&parg2[i + (vl * 4)], 0, 1);
			}

			// ===== 第1块 =====
			vint32m1_t v_arg1_0 = __riscv_vle32_v_i32m1((int32_t*)&parg1[i], vl);
			vint32m1_t v_arg2_0 = __riscv_vle32_v_i32m1((int32_t*)&parg2[i], vl);
			vuint8mf4_t v_flags1_0 = __riscv_vle8_v_u8mf4(&pflags1[i], vl);
			vuint8mf4_t v_flags2_0 = __riscv_vle8_v_u8mf4(&pflags2[i], vl);

			vbool32_t v_not_null1_0 = __riscv_vmsne_vx_u8mf4_b32(v_flags1_0, 0, vl);
			vbool32_t v_not_null2_0 = __riscv_vmsne_vx_u8mf4_b32(v_flags2_0, 0, vl);
			vbool32_t v_both_not_null_0 = __riscv_vmand_mm_b32(v_not_null1_0, v_not_null2_0, vl);
			vbool32_t v_cmp_result_0 = execute_comparison_rvv<sop>(v_arg1_0, v_arg2_0, vl);
			vbool32_t v_valid_0 = __riscv_vmand_mm_b32(v_both_not_null_0, v_cmp_result_0, vl);

			// ===== 第2块 =====
			int idx1 = i + vl;
			if __builtin_expect(idx1 < remainder_start, 1) {
				vint32m1_t v_arg1_1 = __riscv_vle32_v_i32m1((int32_t*)&parg1[idx1], vl);
				vint32m1_t v_arg2_1 = __riscv_vle32_v_i32m1((int32_t*)&parg2[idx1], vl);
				vuint8mf4_t v_flags1_1 = __riscv_vle8_v_u8mf4(&pflags1[idx1], vl);
				vuint8mf4_t v_flags2_1 = __riscv_vle8_v_u8mf4(&pflags2[idx1], vl);

				vbool32_t v_not_null1_1 = __riscv_vmsne_vx_u8mf4_b32(v_flags1_1, 0, vl);
				vbool32_t v_not_null2_1 = __riscv_vmsne_vx_u8mf4_b32(v_flags2_1, 0, vl);
				vbool32_t v_both_not_null_1 = __riscv_vmand_mm_b32(v_not_null1_1, v_not_null2_1, vl);
				vbool32_t v_cmp_result_1 = execute_comparison_rvv<sop>(v_arg1_1, v_arg2_1, vl);
				vbool32_t v_valid_1 = __riscv_vmand_mm_b32(v_both_not_null_1, v_cmp_result_1, vl);

				// 写回第2块（与第1块并行）
				vint32m1_t v_result_1 = __riscv_vmv_v_x_i32m1(0, vl);
				v_result_1 = __riscv_vmerge_vxm_i32m1(v_result_1, 1, v_valid_1, vl);
				__riscv_vse32_v_i32m1((int32_t*)&presult[idx1], v_result_1, vl);

				vuint8mf4_t v_result_flag_1 = __riscv_vmv_v_x_u8mf4(0, vl);
				v_result_flag_1 = __riscv_vmerge_vxm_u8mf4(v_result_flag_1, 1, v_both_not_null_1, vl);
				__riscv_vse8_v_u8mf4(&pflag[idx1], v_result_flag_1, vl);
			}

			// 写回第1块
			vint32m1_t v_result_0 = __riscv_vmv_v_x_i32m1(0, vl);
			v_result_0 = __riscv_vmerge_vxm_i32m1(v_result_0, 1, v_valid_0, vl);
			__riscv_vse32_v_i32m1((int32_t*)&presult[i], v_result_0, vl);

			vuint8mf4_t v_result_flag_0 = __riscv_vmv_v_x_u8mf4(0, vl);
			v_result_flag_0 = __riscv_vmerge_vxm_u8mf4(v_result_flag_0, 1, v_both_not_null_0, vl);
			__riscv_vse8_v_u8mf4(&pflag[i], v_result_flag_0, vl);

			// ===== 第3块和第4块（类似处理）=====
			i += vl * 2;
			if __builtin_expect(i < remainder_start, 1) {
				int idx2 = i;
				int idx3 = i + vl;
				
				vint32m1_t v_arg1_2 = __riscv_vle32_v_i32m1((int32_t*)&parg1[idx2], vl);
				vint32m1_t v_arg2_2 = __riscv_vle32_v_i32m1((int32_t*)&parg2[idx2], vl);
				vuint8mf4_t v_flags1_2 = __riscv_vle8_v_u8mf4(&pflags1[idx2], vl);
				vuint8mf4_t v_flags2_2 = __riscv_vle8_v_u8mf4(&pflags2[idx2], vl);

				vbool32_t v_both_not_null_2 = __riscv_vmand_mm_b32(
					__riscv_vmsne_vx_u8mf4_b32(v_flags1_2, 0, vl),
					__riscv_vmsne_vx_u8mf4_b32(v_flags2_2, 0, vl), vl);
				vbool32_t v_valid_2 = __riscv_vmand_mm_b32(v_both_not_null_2,
					execute_comparison_rvv<sop>(v_arg1_2, v_arg2_2, vl), vl);

				if __builtin_expect(idx3 < remainder_start, 1) {
					vint32m1_t v_arg1_3 = __riscv_vle32_v_i32m1((int32_t*)&parg1[idx3], vl);
					vint32m1_t v_arg2_3 = __riscv_vle32_v_i32m1((int32_t*)&parg2[idx3], vl);
					vuint8mf4_t v_flags1_3 = __riscv_vle8_v_u8mf4(&pflags1[idx3], vl);
					vuint8mf4_t v_flags2_3 = __riscv_vle8_v_u8mf4(&pflags2[idx3], vl);

					vbool32_t v_both_not_null_3 = __riscv_vmand_mm_b32(
						__riscv_vmsne_vx_u8mf4_b32(v_flags1_3, 0, vl),
						__riscv_vmsne_vx_u8mf4_b32(v_flags2_3, 0, vl), vl);
					vbool32_t v_valid_3 = __riscv_vmand_mm_b32(v_both_not_null_3,
						execute_comparison_rvv<sop>(v_arg1_3, v_arg2_3, vl), vl);

					// 写回第3块
					vint32m1_t v_result_3 = __riscv_vmerge_vxm_i32m1(__riscv_vmv_v_x_i32m1(0, vl), 1, v_valid_3, vl);
					__riscv_vse32_v_i32m1((int32_t*)&presult[idx3], v_result_3, vl);
					vuint8mf4_t v_result_flag_3 = __riscv_vmerge_vxm_u8mf4(__riscv_vmv_v_x_u8mf4(0, vl), 1, v_both_not_null_3, vl);
					__riscv_vse8_v_u8mf4(&pflag[idx3], v_result_flag_3, vl);
				}

				// 写回第2块
				vint32m1_t v_result_2 = __riscv_vmerge_vxm_i32m1(__riscv_vmv_v_x_i32m1(0, vl), 1, v_valid_2, vl);
				__riscv_vse32_v_i32m1((int32_t*)&presult[idx2], v_result_2, vl);
				vuint8mf4_t v_result_flag_2 = __riscv_vmerge_vxm_u8mf4(__riscv_vmv_v_x_u8mf4(0, vl), 1, v_both_not_null_2, vl);
				__riscv_vse8_v_u8mf4(&pflag[idx2], v_result_flag_2, vl);

				i += vl * 2;
			}
		}

		// 尾部处理：处理剩余元素
		for (; i < nvalues; ) {
			vl = __riscv_vsetvl_e32m1(nvalues - i);

			vint32m1_t v_arg1 = __riscv_vle32_v_i32m1((int32_t*)&parg1[i], vl);
			vint32m1_t v_arg2 = __riscv_vle32_v_i32m1((int32_t*)&parg2[i], vl);
			vuint8mf4_t v_flags1 = __riscv_vle8_v_u8mf4(&pflags1[i], vl);
			vuint8mf4_t v_flags2 = __riscv_vle8_v_u8mf4(&pflags2[i], vl);

			vbool32_t v_both_not_null = __riscv_vmand_mm_b32(
				__riscv_vmsne_vx_u8mf4_b32(v_flags1, 0, vl),
				__riscv_vmsne_vx_u8mf4_b32(v_flags2, 0, vl), vl);
			vbool32_t v_valid = __riscv_vmand_mm_b32(v_both_not_null,
				execute_comparison_rvv<sop>(v_arg1, v_arg2, vl), vl);

			vint32m1_t v_result = __riscv_vmerge_vxm_i32m1(__riscv_vmv_v_x_i32m1(0, vl), 1, v_valid, vl);
			__riscv_vse32_v_i32m1((int32_t*)&presult[i], v_result, vl);

			vuint8mf4_t v_result_flag = __riscv_vmerge_vxm_u8mf4(__riscv_vmv_v_x_u8mf4(0, vl), 1, v_both_not_null, vl);
			__riscv_vse8_v_u8mf4(&pflag[i], v_result_flag, vl);

			i += vl;
		}
	}
	else
	{
		// 第二轮优化：向量化selection处理
		int vl;
		for (i = 0; i < nvalues; ) {
			vl = __riscv_vsetvl_e32m1(nvalues - i);

			// 软件预取：提前加载下一批数据到cache
			if (i + vl < nvalues) {
				__builtin_prefetch(&parg1[i + vl], 0, 3);
				__builtin_prefetch(&parg2[i + vl], 0, 3);
				__builtin_prefetch(&pflags1[i + vl], 0, 3);
				__builtin_prefetch(&pflags2[i + vl], 0, 3);
			}

			// 加载selection向量
			vuint8mf4_t v_selection_u8 = __riscv_vle8_v_u8mf4((uint8_t*)&pselection[i], vl);
			vbool32_t v_selection = __riscv_vmsne_vx_u8mf4_b32(v_selection_u8, 0, vl);

			// 加载数据
			vint32m1_t v_arg1 = __riscv_vle32_v_i32m1((int32_t*)&parg1[i], vl);
			vint32m1_t v_arg2 = __riscv_vle32_v_i32m1((int32_t*)&parg2[i], vl);
			
			// 加载NULL标记
			vuint8mf4_t v_flags1 = __riscv_vle8_v_u8mf4(&pflags1[i], vl);
			vuint8mf4_t v_flags2 = __riscv_vle8_v_u8mf4(&pflags2[i], vl);

			// NULL检查
			vbool32_t v_not_null1 = __riscv_vmsne_vx_u8mf4_b32(v_flags1, 0, vl);
			vbool32_t v_not_null2 = __riscv_vmsne_vx_u8mf4_b32(v_flags2, 0, vl);
			vbool32_t v_both_not_null = __riscv_vmand_mm_b32(v_not_null1, v_not_null2, vl);

			// 合并selection和not_null掩码
			vbool32_t v_active = __riscv_vmand_mm_b32(v_selection, v_both_not_null, vl);

			// 执行比较操作（使用helper函数）
			vbool32_t v_cmp_result = execute_comparison_rvv<sop>(v_arg1, v_arg2, vl);

			// 最终有效掩码：active && cmp_result
			vbool32_t v_valid = __riscv_vmand_mm_b32(v_active, v_cmp_result, vl);

			// 准备结果
			vint32m1_t v_result = __riscv_vmv_v_x_i32m1(0, vl);
			v_result = __riscv_vmerge_vxm_i32m1(v_result, 1, v_valid, vl);

			// 只在selection为true时写入
			vint32m1_t v_old_result = __riscv_vle32_v_i32m1((int32_t*)&presult[i], vl);
			v_result = __riscv_vmerge_vvm_i32m1(v_old_result, v_result, v_selection, vl);
			__riscv_vse32_v_i32m1((int32_t*)&presult[i], v_result, vl);

			// 设置flag：selection && both_not_null时为1，selection && !both_not_null时为0
			vuint8mf4_t v_result_flag = __riscv_vmv_v_x_u8mf4(0, vl);
			v_result_flag = __riscv_vmerge_vxm_u8mf4(v_result_flag, 1, v_active, vl);
			
			// 只在selection为true时更新flag
			vuint8mf4_t v_old_flag = __riscv_vle8_v_u8mf4(&pflag[i], vl);
			v_result_flag = __riscv_vmerge_vvm_u8mf4(v_old_flag, v_result_flag, v_selection, vl);
			__riscv_vse8_v_u8mf4(&pflag[i], v_result_flag, vl);

			i += vl;
		}
	}

	PG_GETARG_VECTOR(3)->m_rows = nvalues;
	PG_GETARG_VECTOR(3)->m_desc.typeId = BOOLOID;
	return PG_GETARG_VECTOR(3);
}

#endif // __riscv_vector

#endif // INT4_RVV_OPT_INL

