from setuptools import setup
pname='fieldmap'
setup(name=pname,
      version='0.1',
      description='Maps from field fluctuations along the light cone',
      url='http://github.com/exgalsky/fieldmap',
      author='exgalsky collaboration',
      license_files = ('LICENSE',),
      packages=[pname],
      entry_points ={ 
        'console_scripts': [ 
          'xgfieldmap = fieldmap.command_line_interface:main'
        ]
      }, 
      zip_safe=False)
