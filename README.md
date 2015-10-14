# company-lua #

Company-lua is a [company-mode](http://company-mode.github.io/)
completion backend for `Lua`.

We use api files from
[ZeroBrane Studio](https://github.com/pkulchenko/ZeroBraneStudio) as source for
the completion candidates. Right now only Lua 5.1, 5.2, 5.3 and
[LÖVE](https://love2d.org/) are supported.

## Installation ##

### Manual ###

Add `company-lua` to the `load-path`:

```lisp
(add-to-list 'load-path "path/to/company-lua")
```

Add the following to your `init.el`:

```lisp
(require 'company)
(require 'company-lua)
(add-to-list 'company-backends 'company-lua)
```

or if you only want to use `company-lua` as backend:

```lisp
(add-hook 'lua-mode-hook
          (lambda () (setq-local company-backends '(company-lua))))
```
