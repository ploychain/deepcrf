import pokers as pkrs

print("可能的 Rank / Suit 枚举：")
for name in dir(pkrs):
    obj = getattr(pkrs, name)
    if hasattr(obj, "__members__"):
        print(f"{name}: {list(obj.__members__.keys())}")

