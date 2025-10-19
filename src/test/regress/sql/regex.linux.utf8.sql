/*
 * This test is for Linux/glibc systems and others that implement proper
 * locale classification of Unicode characters with high code values.
 * It must be run in a database with UTF8 encoding and a Unicode-aware locale.
 */
SET client_encoding TO UTF8;
--
-- Test the "high colormap" logic with single characters and ranges that
-- exceed the MAX_SIMPLE_CHR cutoff, here assumed to be less than U+2000.
--
-- trivial cases:
SELECT 'aⓕ' ~ 'a[ⓐ-ⓩ]' AS t;
SELECT 'aⒻ' ~ 'a[ⓐ-ⓩ]' AS f;
-- cases requiring splitting of ranges:
SELECT 'aⓕⓕ' ~ 'aⓕ[ⓐ-ⓩ]' AS t;
SELECT 'aⓕⓐ' ~ 'aⓕ[ⓐ-ⓩ]' AS t;
SELECT 'aⓐⓕ' ~ 'aⓕ[ⓐ-ⓩ]' AS f;
SELECT 'aⓕⓕ' ~ 'a[ⓐ-ⓩ]ⓕ' AS t;
SELECT 'aⓕⓐ' ~ 'a[ⓐ-ⓩ]ⓕ' AS f;
SELECT 'aⓐⓕ' ~ 'a[ⓐ-ⓩ]ⓕ' AS t;
SELECT 'aⒶⓜ' ~ 'a[Ⓐ-ⓜ][ⓜ-ⓩ]' AS t;
SELECT 'aⓜⓜ' ~ 'a[Ⓐ-ⓜ][ⓜ-ⓩ]' AS t;
SELECT 'aⓜⓩ' ~ 'a[Ⓐ-ⓜ][ⓜ-ⓩ]' AS t;
SELECT 'aⓩⓩ' ~ 'a[Ⓐ-ⓜ][ⓜ-ⓩ]' AS f;
SELECT 'aⓜ⓪' ~ 'a[Ⓐ-ⓜ][ⓜ-ⓩ]' AS f;
SELECT 'a0' ~ 'a[a-ⓩ]' AS f;
SELECT 'aq' ~ 'a[a-ⓩ]' AS t;
SELECT 'aⓜ' ~ 'a[a-ⓩ]' AS t;
SELECT 'a⓪' ~ 'a[a-ⓩ]' AS f;
-- Locale-dependent character classes
SELECT 'aⒶⓜ⓪' ~ '[[:alpha:]][[:alpha:]][[:alpha:]][[:graph:]]' AS t;
SELECT 'aⒶⓜ⓪' ~ '[[:alpha:]][[:alpha:]][[:alpha:]][[:alpha:]]' AS f;
-- Locale-dependent character classes with high ranges
SELECT 'aⒶⓜ⓪' ~ '[a-z][[:alpha:]][ⓐ-ⓩ][[:graph:]]' AS t;
SELECT 'aⓜⒶ⓪' ~ '[a-z][[:alpha:]][ⓐ-ⓩ][[:graph:]]' AS f;
SELECT 'aⓜⒶ⓪' ~ '[a-z][ⓐ-ⓩ][[:alpha:]][[:graph:]]' AS t;
SELECT 'aⒶⓜ⓪' ~ '[a-z][ⓐ-ⓩ][[:alpha:]][[:graph:]]' AS f;
-- Test about the Chinese punctuation marks
SELECT regexp_replace(
        '1,23`$@……^、，\t#￥！&*SLFDKJ中文',
        '[[:punct:]]',
        '',
        'g'
    );
drop table if exists numtest;
create table numtest(
num int, address varchar);
Insert into numtest values(1,'Paris'),(2,'BJ'),(3,null),(4,'123asd'),(5,'GZ'),(6,'~!@ASDzxc'),(7,'  12_\');
select * from numtest where regexp_like(address,'^[A-J]');
select * from numtest where regexp_like(address,'[[:punct:]]');
select * from numtest where regexp_like(address,'^[^[:digit:]]+$');
drop table numtest;
