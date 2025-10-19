/*
 * Copyright (c) 2021 Huawei Technologies Co.,Ltd.
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
 * dcf_word.h
 *
 *
 * IDENTIFICATION
 *    src/common/lexer/dcf_word.h
 *
 * -------------------------------------------------------------------------
 */

#ifndef __DCF_WORD_H__
#define __DCF_WORD_H__

#include "cm_defs.h"
#include "cm_text.h"
#include "cm_num.h"
#include "dcf_key_word.h"
#include "lang_text.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum en_word_type {
    WORD_TYPE_EOF = 0x00000000,
    WORD_TYPE_UNKNOWN = 0x00000001,
    WORD_TYPE_VARIANT = 0x00000002,
    WORD_TYPE_KEYWORD = 0x00000004,
    WORD_TYPE_OPERATOR = 0x00000008,
    WORD_TYPE_STRING = 0x00000010,
    WORD_TYPE_PARAM = 0x00000020,
    WORD_TYPE_COMPARE = 0x00000040,
    WORD_TYPE_FUNCTION = 0x00000080,
    WORD_TYPE_NUMBER = 0x00000100,
    WORD_TYPE_DATATYPE = 0x00000200,
    WORD_TYPE_BRACKET = 0x00000400,
    WORD_TYPE_SPEC_CHAR = 0x00000800,
    WORD_TYPE_RESERVED = 0x00001000,
    WORD_TYPE_SIZE = 0x00002000,
    WORD_TYPE_COMMENT = 0x00004000,
    WORD_TYPE_STAR = 0x00008000,
    WORD_TYPE_PL_ATTR = 0x00010000,  // SQL%ROWCOUNT, V1%TYPE, TABLE1%ROWTYPE
    WORD_TYPE_ANCHOR = 0x00020000,
    WORD_TYPE_DQ_STRING = 0x00040000,
    WORD_TYPE_PL_TERM = 0x00080000,     // LINE TERMINATOR ';'
    WORD_TYPE_PL_SETVAL = 0x00200000,   // PL/SQL input variant for embedded sql
    WORD_TYPE_PL_RANGE = 0x00400000,    // PL/SQL for statment range spilt like '..'
    WORD_TYPE_PL_NEW_COL = 0x00800000,  // ':NEW.COLUMN' in trigger, for example, ':new.f1'
    WORD_TYPE_PL_OLD_COL = 0x01000000,  // ':OLD.COLUMN' in trigger, for example, ':old.f1'
    WORD_TYPE_HINT_KEYWORD = 0x04000000,
    WORD_TYPE_HEXADECIMAL = 0x08000000,
    WORD_TYPE_JOIN_COL = 0x10000000,  // t1.f1(+) = t2.f1
    WORD_TYPE_ARRAY = 0x20000000, // array, e.g. '{1,2}', array[1,2]
    WORD_TYPE_ERROR = 0x40000000,
} word_type_t;

/* can't change index of reserved!!! */
typedef enum en_reserved_wid {
    RES_WORD_CONNECT_BY_ISCYCLE = SQL_RESERVED_WORD_BASE + 1,
    RES_WORD_CONNECT_BY_ISLEAF,
    RES_WORD_CTID,
    RES_WORD_DEFAULT,
    RES_WORD_DELETING,
    RES_WORD_FALSE,
    RES_WORD_INSERTING,
    RES_WORD_LEVEL,
    RES_WORD_NULL,
    RES_WORD_ROWID,
    RES_WORD_ROWNUM,
    RES_WORD_ROWSCN,
    RES_WORD_SESSIONTZ,
    RES_WORD_SYSDATE,
    RES_WORD_SYSTIMESTAMP,
    RES_WORD_TRUE,
    RES_WORD_UPDATING,
    RES_WORD_USER,
    RES_WORD_DATABASETZ,
    RES_WORD_CURDATE,
    RES_WORD_CURTIMESTAMP,
    RES_WORD_LOCALTIMESTAMP,
    RES_WORD_DUMMY,
    RES_WORD_UTCTIMESTAMP,
    RES_WORD_COLUMN_VALUE,
} reserved_wid_t;

typedef enum en_datatype_wid {
    DTYP_BIGINT = 0,      /* bigint [signed] */
    DTYP_UBIGINT,         /* [ubigint | bigint unsigned] */
    DTYP_DOUBLE,          /* double(M, N) */
    DTYP_FLOAT,           /* float(M, N) */
    DTYP_INTEGER,         /* integer [signed] | int [signed] */
    DTYP_UINTEGER,        /* uint | uinteger | integer unsigned */
    DTYP_SMALLINT,        /* smallint [signed] | short [signed]  */
    DTYP_USMALLINT,       /* usmallint | ushort | [smallint | short] unsigned */
    DTYP_TINYINT,         /* tinyint [signed] */
    DTYP_UTINYINT,        /* utinyint | tinyint unsigned */
    DTYP_NUMBER,          /* number */
    DTYP_DECIMAL,         /* numeric, decimal */
    DTYP_BINARY,          /* binary  */
    DTYP_VARBINARY,       /* varbinary */
    DTYP_RAW,             /* binary  */
    DTYP_BLOB,            /* blob */
    DTYP_CLOB,            /* clob */
    DTYP_BOOLEAN,         /* bool | boolean */
    DTYP_CHAR,            /* char */
    DTYP_VARCHAR,         /* varchar */
    DTYP_SERIAL,          /* serial */
    DTYP_DATE,            /* date | datetime */
    DTYP_TIMESTAMP,       /* timestamp */
    DTYP_TIMESTAMP_TZ,    /* timestamp with time zone */
    DTYP_TIMESTAMP_LTZ,   /* timestamp with local time zone */
    DTYP_INTERVAL,        /* interval */
    DTYP_INTERVAL_DS,     /* interval day to second */
    DTYP_INTERVAL_YM,     /* interval year to month */
    DTYP_BINARY_DOUBLE,   /* binary_double */
    DTYP_BINARY_FLOAT,    /* binary_float */
    DTYP_BINARY_INTEGER,  /* binary_integer [signed] */
    DTYP_BINARY_BIGINT,   /* binary_bigint [signed] */
    DTYP_BINARY_UINTEGER, /* binary_integer unsigned */
    DTYP_BINARY_UBIGINT,  /* binary_bigint unsigned */
    DTYP_NVARCHAR,        /* nvarchar */
    DTYP_NCHAR,           /* nchar */
    DTYP_IMAGE,           /* image, equals to longblob */
    DTYP_ARRAY,           /* array, only used for procedure or self defined function's argument type definition */
    DTYP__SIZE_           /* do not use, representing the number of datatype words */
} datatype_wid_t;

typedef enum en_word_flag {
    WORD_FLAG_NONE = 0x00000000,
    WORD_FLAG_NEGATIVE = 0x00000001,
} word_flag_t;

#define MAX_EXTRA_TEXTS 6

typedef struct st_ex_text {
    word_type_t type;
    lang_text_t text;
} ex_text_t;

typedef struct st_word {
    uint32 id;
    word_type_t type;
    char *begin_addr;
    src_loc_t loc;
    lang_text_t text;
    word_flag_t flag_type;

    union {
        struct {
            uint32 ex_count;
            ex_text_t ex_words[MAX_EXTRA_TEXTS];
            bool32 namable;
            word_type_t ori_type; /* original type of this word, used when change the type of one word */
        };

        // np is used when the word_type is WORD_TYPE_NUMBER or WORD_TYPE_SIZE
        num_part_t np;
    };
} word_t;

typedef struct st_key_word {
    uint32 id;
    bool32 namable;
    text_t text;
} key_word_t;

typedef struct st_datatype_word {
    text_t text;
    datatype_wid_t id;
    bool32 namable;
    bool32 can_sign; /* can has signed or unsigned */
} datatype_word_t;

typedef struct st_column_word {
    lang_text_t user;
    lang_text_t table;
    lang_text_t name;
    int32 ss_start; /* for array type. e.g. t1.f1[2,3], subscript start from 2 */
    int32 ss_end;   /* for array type. e.g. t1.f1[2,3], subscript end by 3 */
} column_word_t;

typedef struct st_func_word {
    lang_text_t user;
    lang_text_t pack;
    lang_text_t name;
    lang_text_t args;
    bool32 user_func_first;  // whether using user function first
} func_word_t;

typedef struct st_table_word {
    lang_text_t user;
    lang_text_t name;
} table_word_t;

typedef struct st_plattr_word {
    lang_text_t block;
    lang_text_t var;
    lang_text_t attr;
} plattr_word_t;

typedef union un_var_word {
    text_t name;
    column_word_t column;
    func_word_t func;
    table_word_t table;
    plattr_word_t plattr;
} var_word_t;

struct st_lex;

status_t lex_match_keyword(struct st_lex *lex, word_t *word);
status_t lex_match_hint_keyword(struct st_lex *lex, word_t *word);
bool32 lex_match_subset(key_word_t *word_set, int32 count, word_t *word);
bool32 lex_match_datetime_unit(word_t *word);
const datatype_word_t *lex_match_datatype_words(const datatype_word_t *word_set, int32 count, const word_t *word);
status_t lex_try_match_datatype(struct st_lex *lex, word_t *word, bool32 *matched);
bool32 lex_check_datatype(word_t *word);

#ifdef __cplusplus
}
#endif

#endif
