#!C:/Users/sandy/Anaconda2\pythonw.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'navigator-updater==0.1.0','gui_scripts','navigator-updater'
__requires__ = 'navigator-updater==0.1.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('navigator-updater==0.1.0', 'gui_scripts', 'navigator-updater')()
    )
