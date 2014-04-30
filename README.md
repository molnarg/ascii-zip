ascii-zip
=========

A deflate compressor that emits compressed data that is in the [A-Za-z0-9] ASCII byte range.

Example
=======

```bash
$ echo 'Hello ASCII world!' >hello
$ ./compress.py --mode raw --output ./hello.infalted ./hello >/dev/null
$ cat ./hello.infalted
D0Up0IZUnnnnnnnnnnnnnnnnnnnUU5nnnnnn3SUUnUUUwCiudIbEAtwwwEt333
G0GDGGDtGptw0GwDDDGtDGDt33333www03333sDdFPsgWWwackSKKaowOWGQ4
```
