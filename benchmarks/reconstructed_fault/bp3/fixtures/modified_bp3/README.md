# Maintained modified-BP3 inputs

These are byte-for-byte copies, not regenerated numerical inputs. The manifest
records their original provenance and SHA-256. Execution requires no files in
the source investigation directories. Keep this directory with the plugin and
the maintained `bp3_modified_fully_frictional.prm`.

`target_cells.txt` supplies the bulk leaf-cell tree; `fault.txt` supplies the
continuous Q1 fault; `prestress.txt` supplies the fixed initial mature background;
`completion.txt` supplies **both** boundary normalization completions.
`seven_step_clock.csv` is used only for the bounded replay, never as an adaptive
continuation controller. No physical input was changed when copying these files.
