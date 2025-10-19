drop table if exists t_multi_node_write_0001;
create table t_multi_node_write_0001(id int) with(segment=on);
begin;
insert into t_multi_node_write_0001 values(1);
insert into t_multi_node_write_0001 values(3);
insert into t_multi_node_write_0001 values(4);
update t_multi_node_write_0001 set id = 99 where id = 3;
delete from t_multi_node_write_0001 where id = 4;
select * from t_multi_node_write_0001;
end;
start transaction;
cursor cursor_multi_node_write_0018 for select relname,reltype,relowner from pg_class order by 1 limit 5;
move forward 1 from cursor_multi_node_write_0018;
fetch forward 3 from cursor_multi_node_write_0018;
close cursor_multi_node_write_0018;
end;
begin;
select * from t_multi_node_write_0001;
insert into t_multi_node_write_0001 values(5);
select * from t_multi_node_write_0001;
end;
insert into t_multi_node_write_0001 values(6);
select * from t_multi_node_write_0001;
create table t1(a int) with(segment=on);
begin;
insert into t1 values(123);
update t1 set a = 'a';
insert into t1 values(12345);
end;
select * from t1;
begin;
rollback;
drop table if exists t01;
begin;
lock table t_multi_node_write_0001;
end;
drop table if exists account;
create table account(id int,transaction_id int,balance float) with(segment=on);
begin;
insert into account(id,transaction_id,balance) values (0,0,100),(0,1,200),(1,0,300),(1,1,300);
savepoint mypoint;
insert into account(id,transaction_id,balance) values (2,1,400),(2,2,500),(3,3,300),(3,4,600);
select * from account order by 1;
rollback to savepoint mypoint;
insert into account(id,transaction_id,balance) values (4,1,700),(4,2,800),(5,3,300),(5,4,900);
commit;
select * from account;
drop table if exists account;
create table account(id int,transaction_id int,balance float) with(segment=on);
begin;
insert into account(id,transaction_id,balance) values (0,0,100),(0,1,200),(1,0,300),(1,1,300);
savepoint mypoint;
insert into account(id,transaction_id,balance) values (2,1,400),(2,2,500),(3,3,300),(3,4,600);
select * from account order by 1;
release savepoint mypoint;
insert into account(id,transaction_id,balance) values (4,1,700),(4,2,800),(5,3,300),(5,4,900);
select * from account;
rollback;
select * from account;

create or replace function func_multi_node_write_0032(tb_name varchar) return varchar is
begin
    execute immediate 'drop table if exists  ' || tb_name || ';';
    execute immediate 'create table  '|| tb_name || '(id int, name varchar(20));';
    execute immediate 'insert into ' || tb_name || ' values (generate_series(1,10), ''test'');';
    execute immediate 'update ' || tb_name || ' set name=''software'' where id = 3;';
    execute immediate 'delete from ' || tb_name || ' where id = 5;';
return tb_name;
end;
/

start transaction;
call func_multi_node_write_0032('t_multi_node_write_0032');
commit;
select * from t_multi_node_write_0032;

drop table t_multi_node_write_0032;
drop function func_multi_node_write_0032;

start transaction;
call func_multi_node_write_0032('t_multi_node_write_0032');
commit;
select * from t_multi_node_write_0032;

create or replace function func_multi_node_write_0032(tb_name varchar) return varchar is
begin
    execute immediate 'drop table if exists  ' || tb_name || ';';
    execute immediate 'create table  '|| tb_name || '(id int, name varchar(20));';
    execute immediate 'insert into ' || tb_name || ' values (generate_series(1,10), ''test'');';
    execute immediate 'update ' || tb_name || ' set name=''software'' where id = 3;';
    execute immediate 'delete from ' || tb_name || ' where id = 5;';
return tb_name;
end;
/

