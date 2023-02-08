"""Initialise package parameters.

.. attention::
    Import this module at the beginning of every module file then the parameters. But parameters will only be initialised once. ::
    
        import UltraMotionCapture.config.param

- :attr:`UltraMotionCapture.output_msg`
    Default as :attr:`True`. If set as :code:`Fault`, no message will be displaced in terminal.
"""
import UltraMotionCapture

try:
    UltraMotionCapture.output_msg
except:
    UltraMotionCapture.output_msg = True