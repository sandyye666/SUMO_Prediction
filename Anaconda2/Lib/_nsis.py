# (c) Continuum Analytics, Inc. / http://continuum.io
# All Rights Reserved
# This file is under the BSD license

# Helper script which is called from within the nsis install process
# on Windows.  The fact that we put this file into the standard library
# directory is merely a convenience.  This way, functionally can easily
# be tested in an installation.

import os
import sys
import traceback
from os.path import abspath, basename, dirname, join, exists

ROOT_PREFIX = sys.prefix

# Install an exception hook which pops up a message box.
# Ideally, exceptions will get returned to NSIS and logged there,
# etc, but this is a stopgap solution for now.
old_excepthook = sys.excepthook

def gui_excepthook(exctype, value, tb):
    try:
        import ctypes, traceback
        MB_ICONERROR = 0x00000010
        title = u'Installation Error'
        msg = u''.join(traceback.format_exception(exctype, value, tb))
        ctypes.windll.user32.MessageBoxW(0, msg, title, MB_ICONERROR)
    finally:
        # Also call the old exception hook to let it do
        # its thing too.
        old_excepthook(exctype, value, tb)
sys.excepthook = gui_excepthook

# If pythonw is being run, there may be no write function
if sys.stdout and sys.stdout.write:
    out = sys.stdout.write
    err = sys.stderr.write
else:
    import ctypes
    OutputDebugString = ctypes.windll.kernel32.OutputDebugStringW
    OutputDebugString.argtypes = [ctypes.c_wchar_p]
    def out(x):
        OutputDebugString('_nsis.py: ' + x)
    def err(x):
        OutputDebugString('_nsis.py: Error: ' + x)

def mk_menus(remove=False):
    try:
        import menuinst
    except ImportError:
        return
    menu_dir = join(ROOT_PREFIX, 'Menu')
    if exists(menu_dir):
        for fn in os.listdir(menu_dir):
            if fn.endswith('.json'):
                shortcut = join(menu_dir, fn)
                try:
                    menuinst.install(shortcut, remove)
                except Exception as e:
                    out("Failed to process %s...\n" % shortcut)
                    err("Error: %s\n" % str(e))
                    err("Traceback:\n%s\n" % traceback.format_exc(20))
                else:
                    out("Processed %s successfully.\n" % shortcut)


def mk_dirs():
    envs_dir = join(ROOT_PREFIX, 'envs')
    if not exists(envs_dir):
        os.mkdir(envs_dir)

    # some cleanup
    try:
        os.unlink(join(ROOT_PREFIX, 'preconda.tar.bz2'))
    except:
        pass


allusers = (not exists(join(ROOT_PREFIX, '.nonadmin')))
out('allusers is %s\n' % allusers)

# This must be the same as conda's binpath_from_arg() in conda/cli/activate.py
PATH_SUFFIXES = ('',
                 os.path.join('Library', 'mingw-w64', 'bin'),
                 os.path.join('Library', 'usr', 'bin'),
                 os.path.join('Library', 'bin'),
                 'Scripts')


def remove_from_path(root_prefix=None):
    from _system_path import (remove_from_system_path,
                              broadcast_environment_settings_change)

    if root_prefix is None:
        root_prefix = ROOT_PREFIX
    for path in [os.path.normpath(os.path.join(root_prefix, path_suffix))
                 for path_suffix in PATH_SUFFIXES]:
        remove_from_system_path(path, allusers)
    broadcast_environment_settings_change()


def add_to_path(pyversion, arch):
    from _system_path import (add_to_system_path,
                              get_previous_install_prefixes,
                              broadcast_environment_settings_change)

    # If a previous Anaconda install attempt to this location left remnants,
    # remove those.
    remove_from_path(ROOT_PREFIX)

    # If a previously registered Anaconda install left remnants, remove those.
    old_prefixes = get_previous_install_prefixes(pyversion, arch, allusers)
    for prefix in old_prefixes:
        out('Removing old installation at %s from PATH (if any entries get found)\n' % (prefix))
        remove_from_path(prefix)

    # add Anaconda to the path
    add_to_system_path([os.path.normpath(os.path.join(ROOT_PREFIX, path_suffix))
                        for path_suffix in PATH_SUFFIXES], allusers)
    broadcast_environment_settings_change()


def main():
    global ROOT_PREFIX

    if (basename(sys.prefix) == '_conda_' and
              basename(dirname(sys.prefix)) == 'envs'):
        ROOT_PREFIX = abspath(join(sys.prefix, '..', '..'))

    cmd = sys.argv[1].strip()
    if cmd == 'mkmenus':
        mk_menus(remove=False)
    elif cmd == 'rmmenus':
        mk_menus(remove=True)
    elif cmd == 'mkdirs':
        mk_dirs()
    elif cmd == 'addpath':
        # These checks are probably overkill, but could be useful
        # if I forget to update something that uses this code.
        if len(sys.argv) > 2:
            pyver = sys.argv[2]
        else:
            pyver = '%s.%s.%s' % (sys.version_info.major,
                                  sys.version_info.minor,
                                  sys.version_info.micro)
        if len(sys.argv) > 3:
            arch = sys.argv[2]
        else:
            arch = '32-bit' if tuple.__itemsize__==4 else '64-bit'
        add_to_path(pyver, arch)
    elif cmd == 'rmpath':
        remove_from_path()
    else:
        sys.exit("ERROR: did not expect %r" % cmd)


if __name__ == '__main__':
    main()
