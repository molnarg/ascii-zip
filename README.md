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

Why?
====

Primarily for crafting ASCII Flash files. Why is this good? This algorithm is at the heart of the Rosetta Flash vulnerability (see [Abusing JSONP with Rosetta Flash](http://miki.it/blog/2014/7/8/abusing-jsonp-with-rosetta-flash/) and [Bypassing Same Origin Policy With JSONP APIs and Flash](https://hackerone.com/reports/10373)).
