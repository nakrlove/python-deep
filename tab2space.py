import sys

src = sys.argv[1]
dst = sys.argv[2]


# with open("word_tab.txt","r") as f:
with open(src,"r") as f:    
    data = f.read()
    data = data.replace("\t","  ")
    
    # with open("word_tab1.txt","w") as fw:
    with open(dst,"w") as fw:
        fw.write(data)