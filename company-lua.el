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


(defvar company-lua-complete-script
  (f-join (f-dirname (or buffer-file-name load-file-name)) "lua/complete.lua")
  "Script file for completion.")

(defun company-lua--parse-output (prefix)
  "Parse output of `company-lua-complete-script' for PREFIX."
  (goto-char (point-min))
  (let ((pattern "\\(.*\\),\\(.*\\),\\(.*\\)$")
        (case-fold-search nil)
        result)
    (while (re-search-forward pattern nil t)
      (let ((item (match-string-no-properties 1))
            (kind (match-string-no-properties 2))
            (desc (match-string-no-properties 3))
            word)
        (when (string-prefix-p prefix item)
          (setq word (substring-no-properties item))
          (push (propertize word 'kind kind 'meta desc) result))))
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
                            "lua" args)))
        (set-process-sentinel
         process
         (lambda (proc status)
           (funcall
            callback
            (let ((res (process-exit-status proc)))
              (with-current-buffer buf
                (company-lua--parse-output prefix))))))))))

(defun company-lua--build-args ()
  (list company-lua-complete-script))

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
  (let ((annotation (get-text-property 0 'kind candidate)))
    (when annotation
      (concat " [" annotation "]"))))

(defun company-lua--meta (candidate)
  (get-text-property 0 'meta candidate))

(defun company-lua--prefix ()
  (when (and lua-mode
             (not (company-in-string-or-comment)))
    (with-syntax-table (copy-syntax-table)
      (modify-syntax-entry ?. "_")
      (modify-syntax-entry ?: "_")
      (or (company-grab-symbol-cons "\\." 1)
          'stop))))

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
