import re
data = """
park 800905-1049118
kim  700905-1059119
"""


pf = re.compile(r"-(\w+)")
result = pf.sub(r"*******", data)
print(result)



s = '<html><head><title>Title</title>'
print(re.match('<.*>', s).span())
print(re.match('<html|HTML.*>', s).group())
