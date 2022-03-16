def import_file(full_name, path):
    """Import a python module from a path. 3.4+ only.
    Does not call sys.modules[full_name] = path
    Credit: https://stackoverflow.com/questions/27381264/python-3-4-how-to-import-a-module-given-the-full-path
    """
    from importlib import util

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)

    spec.loader.exec_module(mod)
    return mod
