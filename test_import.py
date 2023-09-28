from test_module import TestModule, new_func

if __name__ == '__main__':
  try:
    tm = TestModule()
    tm.test_func()
    new_func()
  except Exception as ex:
    print(ex)