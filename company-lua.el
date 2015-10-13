;;; company-lua.el --- Company backend for Lua       -*- lexical-binding: t; -*-

;; Copyright (C) 2015  Peter Vasil

;; Author: Peter Vasil <mail@petervasil.net>
;; Keywords:

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <http://www.gnu.org/licenses/>.

;;; Commentary:

;;

;;; Code:

(require 'lua-mode)
(require 'f)
(require 's)


(defgroup company-lua nil
  "Completion backend for Lua."
  :group 'company)

(defcustom company-lua-executable
  (executable-find "lua")
  "Location of Lua executable."
  :type 'file
  :group 'company-lua)

(defvar company-lua-complete-script
  (f-join (f-dirname (f-this-file)) "lua/complete.lua")
  "Script file for completion.")

(defun company-lua--parse-output (prefix)
  "Parse output of `company-lua-complete-script' for PREFIX."
  (goto-char (point-min))
  (let ((pattern "word:\\(.*\\),kind:\\(.*\\),args:\\(.*\\),returns:\\(.*\\),doc:\\(.*\\)$")
        (case-fold-search nil)
        result)
    (while (re-search-forward pattern nil t)
      (let ((item (match-string-no-properties 1))
            (kind (match-string-no-properties 2))
            (args (match-string-no-properties 3))
            (returns (match-string-no-properties 4))
            (doc (match-string-no-properties 5)))
        (push (propertize item 'kind kind 'args args 'returns returns 'doc doc) result)))
    result))

(defun company-lua--start-process (prefix callback &rest args)
  (let ((buf (get-buffer-create "*company-lua-output*"))
        (process-adaptive-read-buffering nil))
    (if (get-buffer-process buf)
        (funcall callback nil)
      (with-current-buffer buf
        (erase-buffer)
        (setq buffer-undo-list t))
      (let ((process (apply #'start-process "lua" buf
                            company-lua-executable args)))
        (set-process-sentinel
         process
         (lambda (proc status)
           (funcall
            callback
            (let ((res (process-exit-status proc)))
              (with-current-buffer buf
                (company-lua--parse-output prefix))))))))))

(defun company-lua--build-args ()
  (list company-lua-complete-script (lua-funcname-at-point)))

(defun company-lua--get-candidates (prefix callback)
  (apply 'company-lua--start-process
         prefix
         callback
         (company-lua--build-args)))

(defun company-lua--candidates (prefix)
  "Candidates handler for the company backend."
  (cons :async (lambda (cb)
                 (company-lua--get-candidates prefix cb))))

(defun company-lua--annotation (candidate)
  (let ((kind (get-text-property 0 'kind candidate))
        (returns (get-text-property 0 'returns candidate))
        (args (get-text-property 0 'args candidate)))
    (concat
     (when (s-present? args) args)
     (when (s-present? returns) (s-prepend " -> " returns))
     (when (s-present? kind) (format " [%s]" kind)))))

(defun company-lua--meta (candidate)
  (get-text-property 0 'doc candidate))

(defun company-lua--prefix ()
  (unless (company-in-string-or-comment)
    (or (company-grab-symbol-cons "\\." 1)
        'stop)))

(defun company-lua (command &optional arg &rest ignored)
  "`company-mode' completion back-end for Lua."
  (interactive (list 'interactive))
  (cl-case command
    (interactive (company-begin-backend 'company-lua))
    (prefix (company-lua--prefix))
    (candidates (company-lua--candidates arg))
    (annotation (company-lua--annotation arg))
    (meta (company-lua--meta arg))
    (duplicates t)))

(provide 'company-lua)
;;; company-lua.el ends here
