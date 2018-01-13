/*
Studio: a simple GUI for the libfive CAD kernel
Copyright (C) 2017  Matt Keeter

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
#pragma once

#include <QApplication>
#include <QMainWindow>
#include <QMessageBox>

class Editor;
class View;
struct Documentation;
namespace Kernel { class Mesh; }

class Window : public QMainWindow
{
    Q_OBJECT
public:
    /*
     *  If target is the null QString, then loads the tutorial file
     *  on first run.
     */
    explicit Window(QString target);

    /*
     *  Loads a file by path, properly checking if the existing document
     *  is unsaved and asking the user to save it in that case.
     */
    void openFile(const QString& name);

protected slots:
    void onOpen(bool=false);
    void onRevert(bool=false);
    bool onSave(bool=false);
    bool onSaveAs(bool=false);
    void onNew(bool=false);
    void onExport(bool=false);
    void onAbout(bool=false);
    void onLoadTutorial(bool=false);
    void onShowDocs(bool=false);

    void onExportReady(QList<const Kernel::Mesh*> shapes);
    void setDocs(Documentation* docs);

signals:
    void exportDone();

protected:
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
    void closeEvent(QCloseEvent* event) override;

    QMessageBox::StandardButton checkUnsaved();
    void setFilename(const QString& str);
    QString workingDirectory() const;

    bool loadFile(QString f);
    bool saveFile(QString f);

    /*  Filename of the current file, or empty string */
    QString filename;

    /*  Used to store the export target while meshes are being generated
     *  and the main event loop is blocked by a progress dialog */
    QString export_filename;

    Editor* editor;
    View* view;
    bool closing=false;
};
