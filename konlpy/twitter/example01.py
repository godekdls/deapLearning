from konlpy.tag import Twitter

twitter = Twitter()
malist = twitter.pos("우유니 사막에 가고싶은데 너무 멀어요 ㅠㅠ", norm=True, stem=True)
print(malist)