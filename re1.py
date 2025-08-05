data = """
park 800905-1049118
kim  700905-1059119
"""

# result = []
# for line in data.split("\n"):
#     word_result = []
#     for word in line.split(" "):
#         if len(word) == 14 and word[:6].isdigit() and word[7:].isdigit():
#             word = word[:6] + "-" + "*******"
#         word_result.append(word)
#     result.append(" ".join(word_result))
# print("\n".join(result))


import re
pat = re.compile("(\d{6})[-]\d{7}")
print(pat)
print(pat.sub("\g<1>-*******", data))

pat1 = re.compile("\d{7}")
print(pat1)
print(pat1.sub("********", data))



li = ["", "a","ab","a.b","acb","a0b","1", "12", "123", "1234", "abc", '1a' ,' ?#!34a']
# [a-zA-Z]
for i in li :
    print(re.sub("\\w","",i))   # 모든문자 
    print(re.sub("[ac]","*",i))   # a 또는 c
    print(re.sub("[^ac]","*",i))   # a 또는 c 아닌것
    print(re.sub("[a-z]","*",i))
    print(re.sub("[23]$","*",i))
    print(re.sub("[0-9][23]","*",i))
    print(re.sub("[34]","*",i))
    print(re.sub("[34]","*",i))
    
    
for i in li:
    print(re.sub("a.b","*",i))  # a + 모든문자 하나 + b
    print(re.sub("\\d","*",i))  #모든숫자들
    print(re.sub("\\D","*",i))  # 모든 숫자 아닌 것
    
    
# 반복 
# * : 0번 이상 {0,}
# + : 1번 이상 {1,}
# ? : 1번 이하 {0,1}
# {m,n} 

li = ['a','aab','aabb','aaabccc']

for i in li:
    print(re.sub("c{3}","*",i))
    print(re.sub("a{2}","*",i))
    print(re.sub("a{2,2}","*",i))
    print(re.sub("a{2,3}","*",i))
    print(re.sub("a?","*",i))
    print(re.sub("a+","*",i))
    print(re.sub("a*","*",i))
    

phonnum = "010-2222-3333"    
# pat = re.compile("(\d{3})[-](\d{4})[-](\d{4})")  
pat = re.compile("010[-./]\d{4}[-./]\d{4}") 
print(pat.sub("****", phonnum))




text = """
    Please contact us at info@example.com and 
    support@mail.example.co.uk
    """
  #  r'[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+'
pat = re.compile("\w+[@]\w+[.]$")    
print(pat.sub("OO",text))
emails = re.findall(r'[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+', text)
print(emails)


text = 'python Hello PYTHON Hello'
pat = re.compile("[a-z]+")
pat.match(text)
pat.search(text)


pat = re.compile("H[a-z]+")
pat = re.compile("[A-Z]{2,}")
pat.match(text)
pat.search(text)


pat.match(text).group()
s = pat.search(text)
print(s.group())
print(s.start())
print(s.end())
print(s.span())


text = "life is short. You need python"
# pat = re.compile("[a-z]+")
pat = re.compile(r"\b[a-z]{2,}")
pat.search(text)

pat.finditer(text)


# iterable : 순환가능한 , 반복가능한, for 문이 가능한
# iterable 한 데이터: 리스트, 튜플, 딕셔너리, 집합
# iterator : 이터레이터,

for i in pat.finditer(text):
    print(i.group())



text = """life is short.
life is short.
You need python
You need python
"""
# pat = re.compile("life|You")
pat = re.compile("^life",re.M)
pat = re.compile("python$",re.M)
pat = re.compile(r"short\.$",re.M)
pat.sub("***",text)
print(pat.findall(text))


data ="""
park 800905-1049114
park 800905-1049115
park 800905-1049116
park 800905-1049117
"""
data ="""
park 010-2232-2322
kim 010-2255-2332
choi 010-2345-2652
lee 010-8009-1049

"""

#group으
p = re.compile('(ABC)+')
m = p.search('ABCABCABC OK?')

# pat = re.compile(r"\w+\s+\d+[-]\d+[-]\d+")

# pat = re.compile(r"(\w+)\s+((\d+)[-]\d+[-]\d+)")
pat = re.compile(r"(?P<famliy>\w+)\s+((?P<numb>\d+)[-]\d+[-]\d+)")
m = pat.search(data)
m = pat.finditer(data)
for i in m:
    print(i.group(1))

print(m)
print(m.group(0))
print(m.group(1))
print(m.group("famliy"))
print(m.group(2))
print(m.group("numb"))
print(m.group(3))

for i in pat.findall(data):
    print(i)



p = re.compile(r'(\b\w+)\s+\1')
res= p.search('Paris in the the spring').group()
print(res)



address = """
https://wikidocs.net/4308
https://dev-skill.tistory.com/195
https://example.com/page?query=618
"""

pat = re.compile(r"\w+://")
r = pat.sub("*****",address)
r = pat.sub("",address)
print(r)
m = pat.search(address)



 p = re.compile("\w+(?=:)")

 p = re.compile(".+:")
 p = re.compile(".+(?=:)")
 m = p.search("http://google.com")
 print(m.group())


print(re.__doc__)


!pip list | findstr pandas




import pandas as pd
pd.Series([1,2,3,4,5])
s = pd.Series([1,2,3,4,5])
type(s)
sum(s)
s.sum()

print(s)
df = pd.DataFrame([[1,2,3,4,5],[1,2,3,4,5]])
type(df)

# df =pd.read_csv('C:\\Users\\Admin\\study01\\data\\data\\gapminder.tsv',sep='\t')
#df =pd.read_csv('https://github.com/EasysPublishing/do_it_pandas/tree/main/data/gapminder.tsv',sep='\t')
df =pd.read_csv('gapminder.tsv',sep='\t')
print(df)
print(df['pop'])
df['pop'].max()
df.shape
row,column =  df.shape
df.shape[0]
df.shape[1]
df.dtypes
df.info()
df.head()
df.tail(15)

# 열 뽑기
df['pop']
df[['pop']]
df[['pop','year']]
df[['country','continent','year','pop']]
lp = df[['country','continent','year','pop']]
print(lp)

df.pop
df.year


# 행 뽑기 loc,iloc
# loc는 문자
# iloc는 숫자
# 한개 뽑기
df.loc[0,'year']
df.loc[0,['year','country']]
# 
df.iloc[0,2]
df.iloc[0,[3,5]]
df.iloc[[0,4],[3,5]]
df.iloc[[0,4], : ]
df.iloc[ : , [2,4,-1] ]
df

df.iloc[ :20 , 3: ]
글자를 str 이라 하지 않고 object라고 한다



df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9],
    'D': [10, 11, 12],
    'E': [13, 14, 15]
})
df.iloc[ : , [2,-1] ]
