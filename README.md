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
```

Since this backend only gives completion results for lua keywords it might be
good to use `company-lua` in combination with other backends instead adding it
to `company-backends` as single backend.

```lisp
(defun my-lua-mode-company-init ()
  (setq-local company-backends '((company-lua
                                  company-etags
                                  company-dabbrev-code
                                  company-yasnippet))))
(add-hook 'lua-mode-hook #'my-lua-mode-company-init)
```

See documentation of `company-backends`.
