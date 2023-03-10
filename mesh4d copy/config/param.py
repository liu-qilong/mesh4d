"""Initialise package parameters.

.. attention::
    Import this module at the beginning of every module file then the parameters. But parameters will only be initialised once. ::
    
        import mesh4d.config.param

- :attr:`mesh4d.output_msg`
    Default as :attr:`True`. If set as :code:`Fault`, no message will be displaced in terminal.
"""
import mesh4d

try:
    mesh4d.output_msg
except:
    mesh4d.output_msg = True