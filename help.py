import sys
import textwrap
from src.utils.schemas import schedule_schema


def find_prop(schema, prop_name):
    if schema.get('type') == 'object':
        for key, value in schema.get('properties', {}).items():
            if key == prop_name:
                return value
            sub_search = find_prop(value, prop_name)
            if sub_search:
                return sub_search
    elif schema.get('type') == 'array':
        return find_prop(schema.get('items', {}), prop_name)


def print_schema_info(schema, prefix, name, required=False):
    schema_type = schema.get('type', 'any')
    title = schema.get('title', 'No title')
    req = '*' if required else ''

    sub_prefix = '- ' if prefix else ''
    child_prefix = prefix + '   | ' if prefix else ' | '

    # Prints schema header
    print("{}{}{}{} ({}) {}".format(prefix, sub_prefix, name, req, schema_type, title))

    # Print description of the current root schema
    if not prefix and schema.get('description'):
        print("\nDescription :")
        print(textwrap.dedent(schema.get('description')).strip())
        print("")

    # Prints properties schemas
    if schema_type == 'object':
        required_props = schema.get('required', [])
        for key, value in sorted(schema.get('properties', {}).items()):
            req_child = key in required_props
            print_schema_info(value, child_prefix, key, req_child)
    # Prints items schema
    elif schema_type == 'array':
        # If the root element is an array, print its items as root
        if not prefix:
            print_schema_info(schema.get('items', {}), '', '[Items]')
        else:
            print_schema_info(schema.get('items', {}), child_prefix, '[Items]')



if __name__ == "__main__":
    print((
        "Schema description. Execute this this script with no parameters to get a basic description of the\n"
        "full expected `schedule_schema` or pass the name of any property to get it's detailed description.\n"))

    prop_name = sys.argv[1] if len(sys.argv) > 1 else None
    if not prop_name:
        print_schema_info(schedule_schema, '', 'schedule_schema')
    else:
        elem = find_prop(schedule_schema, prop_name)
        if not elem:
            print("'{}' was not found.".format(prop_name))
        else:
            print_schema_info(elem, '', prop_name)
    print("")
