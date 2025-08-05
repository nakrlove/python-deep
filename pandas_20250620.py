import pandas as pd

# df =pd.read_csv('C:\\Users\\Admin\\study01\\data\\data\\gapminder.tsv',sep='\t')
#df =pd.read_csv('https://github.com/EasysPublishing/do_it_pandas/tree/main/data/gapminder.tsv',sep='\t')
df =pd.read_csv('gapminder.tsv',sep='\t')
df.iloc[ 2:5 ]

# 같은값 출력됨
# 열
df.iloc[ :, 1:4 ]
df.loc[ : ,['continent','year','lifeExp']]

# 행
df.iloc[ 1:4 ]
df.iloc[ 1:4 , : ]

df['year']
df[['continent','year','lifeExp']]

# 행 한개
df[0:1]
df['country']

# df.iloc[::step, ::step]

# 정규 표현식
text = ["920213-1456234", "890202-2152341", "880909-2723415", "901208-1551208", "911101-1502891"]

import re
pat = re.compile(r"\w+[-](\d{1})")

a = (text)
a
li = []
for i in text:
    if pat.search(i).group(1) == '1':
       li.append('남자') 
    elif pat.search(i).group(1) == '2':
       li.append('여자') 
       
    

print(li)       

pattern = r"-(1|2)"
# result = [re.sub(pattern, lambda m: "-남자" if m.group(1) == "1" else "-여자", s) for s in text]
result = [("남자" if re.search(pattern, s).group(1) == "1" else "여자") for s in text]
result


text = """
life is short.
life is short.
You need python
You need python.  
"""

# pat = re.compile("life|You")
pat = re.compile("^life",re.M)
pat = re.compile("python$",re.M)
pat = re.compile(r"short\.$",re.M)

pat = re.compile(r"\w{3}[.]?\s*$",re.M)
print(pat.sub("***",text))


pat.sub("***",text)
print(pat.findall(text))


# p = re.compile(r'(\b\w+)\s+\1')
p = re.compile(r'\b\w+\b$')
res= p.search('life is short.').group()

print(res)


pattern = r"-(1|2)"
pat = re.compile("(w+)$", re.M)
pat.search(1)
re.search(pattern, s).group(1)




text = "life is short."

# 마지막 단어만 매칭하는 정규식: 마지막 단어 끝에 문장부호가 있으면 포함되지 않음
pattern = r'\b\w+\b$'
pattern = r'\b\w+[^\w\s]*$'

# re.sub 한 번에 처리
result = re.sub(
    pattern,
    lambda m: (m.group()[:-3] + "***") if len(m.group()) > 3 else "***",
    text
)
print(text)
print(result)

# 휴대폰 번호 가운데 숫자4개를 ****로 바꾸어 보세요
# l로 시작하는 성씨만 가운데 4개를 ****로 바꾸어 봅시다.
text ="""
park 010-2580-2322
lee 010-3355-2332
choi 010-2345-2652
lee 010-8009-1049

"""

# r"-(2580|3355)-1"
# pattern = r"-(1|2)"
pat = re.compile(r"[-](2|3)\d{3}[-]")
pat = re.compile(r"-(2580|3355)-", re.M)
pat = re.compile(r"^(l\w+\s+\d{3}-)(\d{4})(-\d{4})", re.M)
pat.sub(r"\1****\3", text)


pat = re.compile(r"(^l.*)-\d{4}-", re.M)
print(pat.sub(r"\g<1>-****-", text))




s = '<html><head><title>Title</title>'
pat = re.compile("<.+?>")
print(pat.match(s).group())



text = """
https://google.com
https://naver.com
fpt://kakao.com
"""
#  greedly 
# 전방탐색 , 후방탐색

# 전방탐색 
pat = re.compile(r"\w+(?=:)")
pat.findall(text)
# 후방탐색
pat = re.compile(r"(?<=//).+")
pat.findall(text)


text= 'Kakao <p>ryan2!</p> keep a straight face.'
# pat = re.compile(r":.+")

pat = re.compile(r"<p>\w+")
pat = re.compile(r"<p>.+")
pat = re.compile(r"(?<=<p>).+")
pat = re.compile(r".+(?=</p>)")
pat = re.compile(r"(?<=<p>)\w+")

pat = re.compile(r"(?<=<p>).+(?=</p>)")

print(pat.search(text).group())
