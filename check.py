import polars as pl

from str_fund_max_1 import StrFundMax1

strs = []
strs.append(StrFundMax1())
sub_list = []


for strategy in strs:
    strategy.load()
    sub_list += strategy.sub_list

sub_list = list(set(sub_list))
print(sub_list)
