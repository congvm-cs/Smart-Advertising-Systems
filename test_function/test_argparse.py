import argparse


def show_something(dummy):
    print("{}".format(dummy))
    print('hello')


FUNCTION_MAP = {'show_something': show_something}

parser = argparse.ArgumentParser()
parser.add_argument("--show_something", help='list all available apps', choices=FUNCTION_MAP.keys(), action='store')

args = parser.parse_args()

fun = FUNCTION_MAP[args.show_something]
fun()