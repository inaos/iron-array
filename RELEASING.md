## Release Procedure

* Make sure that the new release number is correctly stated in main CMakeLists.txt.

* Merge `develop` branch into `master` via a PR that should be approved at least by 2 people.

* Change into the `master` branch and tag the release using the next convention:

  $ git tag vX.Y.Z -m"Tagging vX.Y.Z release"

* Manually trigger the release pipeline in Azure web interface.

* Check that the new release artifacts appear in the repository (see below).

## Post-release Procedure

* Go to the `develop` branch again and increment the release version info from
  X.Y.Z -> X.Y.Z+1.

* Commit with:

```
  git commit -a -m"Post vX.Y.Z release actions done"
```

### Artifacts Repository

Right now we are using this:

* JFrog Artefactory: https://inaos.jfrog.io

We don't plan to sell the C library, but if in the future some customer wants access to the C library, we could copy these artifacts somwhere, like e.g.:

* https://downloads.ironarray.io
