from setuptools import setup
pname='xgfield'
setup(name=pname,
      version='0.2',
      description='Mocks from field representation of LSS along the light cone',
      url='http://github.com/exgalsky/xgfield',
      author='exgalsky collaboration',
      license_files = ('LICENSE',),
      packages=[pname],
      entry_points ={ 
        'console_scripts': [ 
          'xgfield = xgfield.command_line_interface:main'
        ]
      }, 
      zip_safe=False)
