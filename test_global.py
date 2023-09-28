GLOBAL_FLAG = False

def test_global_func():
    global GLOBAL_FLAG
    if GLOBAL_FLAG is False:
        print("flag is False! will be True...")
        GLOBAL_FLAG = True
    else:
        print("flag is True...")

def global_switch(data: bool):
    global GLOBAL_FLAG
    print("flag will switch ", data)
    GLOBAL_FLAG = data