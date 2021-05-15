import sys
sys.path.append('..')
from datasets.text.numbers import normalize_numbers

if __name__ == '__main__':
    num = '12345.123'
    ch = normalize_numbers(num)
    print(ch)

    num = '12/123'
    ch = normalize_numbers(num)
    print(ch)
