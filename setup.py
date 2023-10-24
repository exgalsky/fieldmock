from setuptools import setup
pname='fieldmock'
setup(name=pname,
      version='0.1',
      description='Mocks from field representation of LSS along the light cone',
      url='http://github.com/exgalsky/fieldmock',
      author='exgalsky collaboration',
      license_files = ('LICENSE',),
      packages=[pname],
      entry_points ={ 
        'console_scripts': [ 
          'xgfieldmock = fieldmock.command_line_interface:main'
        ]
      }, 
      zip_safe=False)
