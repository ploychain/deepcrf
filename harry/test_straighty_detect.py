import pokers as pkrs

def detect_card_enums():
    for name in dir(pkrs):
        obj = getattr(pkrs, name)
        if hasattr(obj, "__members__"):
            print(f"{name}: {[m for m in obj.__members__.keys()]}")
    print("\n示例可用构造方式:")
    print("pkrs.Card( rank_enum_value, suit_enum_value )")

if __name__ == "__main__":
    detect_card_enums()
