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
#include <iostream>

#include <QEvent>
#include <QKeyEvent>
#include <QPointer>
#include <QLabel>
#include <QCompleter>
#include <QTextBrowser>
#include <QVBoxLayout>

#include "studio/documentation.hpp"

void Documentation::insert(QString module, QString name, QString doc)
{
    docs[module][name].doc = doc;
}

////////////////////////////////////////////////////////////////////////////////

QScopedPointer<Documentation> DocumentationPane::docs;
QPointer<DocumentationPane> DocumentationPane::instance;

DocumentationPane::DocumentationPane()
    : search(new QLineEdit)
{
    Q_ASSERT(docs.data());

    // Flatten documentation into a single-level map
    QMap<QString, QString> fs;
    QMap<QString, QString> tags;
    QMap<QString, QString> mods;
    for (auto mod : docs->docs.keys())
    {
        for (auto f : docs->docs[mod].keys())
        {
            fs[f] = docs->docs[mod][f].doc;
            tags.insert(f, "i" + QString::fromStdString(
                        std::to_string(tags.size())));
            mods.insert(f, mod);
        }
    }

    int max_name = 0;
    for (auto& f : fs.keys())
    {
        max_name = std::max(f.length(), max_name);
    }

    // Unpack documentation into a text box
    auto txt = new QTextBrowser();
    for (auto f : fs.keys())
    {
        const auto doc = fs[f];

        auto f_ = doc.count(" ") ? doc.split(" ")[0] : "";

        // Add padding so that the module names all line up
        QString padding;
        for (int i=0; i < max_name - f.length() + 4; ++i)
        {
            padding += "&nbsp;";
        }

        txt->insertHtml(
                "<tt><a name=\"" + tags[f] +
                "\" href=\"#" + tags[f] + "\">" + f + "</a>" +
                padding +
                "<font color=\"silver\">" + mods[f] + "</font>" +
                "</tt><br>");
        if (f_ != f)
        {
            txt->insertHtml("<tt>" + f + "</tt>");
            txt->insertHtml(": alias for ");
            if (fs.count(f_) != 1)
            {
                std::cerr << "DocumentationPane: missing alias "
                          << f_.toStdString() << " for " << f.toStdString()
                          << std::endl;
                txt->insertHtml("<tt>" + f_ + "</tt> (missing)\n");
            }
            else
            {
                txt->insertHtml("<tt><a href=\"#" + tags[f_] + "\">" + f_ + "</a></tt><br>");
            }
            txt->insertPlainText("\n");
        }
        else
        {
            auto lines = doc.split("\n");
            if (lines.size() > 0)
            {
                txt->insertHtml("<tt>" + lines[0] + "</tt><br>");
            }
            for (int i=1; i < lines.size(); ++i)
            {
                txt->insertPlainText(lines[i] + "\n");
            }
            txt->insertPlainText("\n");
        }
    }
    {   // Erase the two trailing newlines
        auto cursor = QTextCursor(txt->document());
        cursor.movePosition(QTextCursor::End);
        cursor.deletePreviousChar();
        cursor.deletePreviousChar();
    }
    txt->setReadOnly(true);
    txt->scrollToAnchor("#i1");
    txt->installEventFilter(this);

    {
        int max_width = 0;
        QFontMetrics fm(txt->font());
        for (auto line : txt->toPlainText().split("\n"))
        {
            max_width = std::max(max_width, fm.width(line));
        }
        txt->setMinimumWidth(max_width + 40);
    }

    // Build a search bar
    auto completer = new QCompleter(fs.keys());
    completer->setCaseSensitivity(Qt::CaseInsensitive);
    search->setCompleter(completer);
    connect(completer, QOverload<const QString&>::of(&QCompleter::highlighted),
            txt, [=](const QString& str){
                if (tags.count(str))
                {
                    txt->scrollToAnchor(tags[str]);
                }
            });
    connect(search, &QLineEdit::textChanged, txt, [=](const QString& str){
                for (auto& t : tags.keys())
                {
                    if (t.startsWith(str))
                    {
                        txt->scrollToAnchor(tags[t]);
                        return;
                    }
                }
            });
    search->installEventFilter(this);

    auto layout = new QVBoxLayout();
    auto row = new QHBoxLayout;
    layout->addWidget(txt);
    row->addSpacing(5);
    row->addWidget(new QLabel("🔍  "));
    row->addSpacing(5);
    row->addWidget(search);
    layout->addItem(row);
    layout->setMargin(0);
    layout->setSpacing(0);
    setLayout(layout);

    setWindowTitle("Shape reference");
    setWindowFlags(Qt::Tool | Qt::CustomizeWindowHint |
                   Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
    setAttribute(Qt::WA_DeleteOnClose);

    show();
    search->setFocus();
}

void DocumentationPane::setDocs(Documentation* ds)
{
    docs.reset(ds);
}

bool DocumentationPane::hasDocs()
{
    return docs.data();
}

void DocumentationPane::open()
{
    if (instance.isNull())
    {
        instance = new DocumentationPane();
    }
    else
    {
        instance->show();
        instance->search->setFocus();
    }
}
bool DocumentationPane::eventFilter(QObject* object, QEvent* event)
{
    Q_UNUSED(object);

    if (event->type() == QEvent::KeyPress &&
        static_cast<QKeyEvent*>(event)->key() == Qt::Key_Escape)
    {
        deleteLater();
        return true;
    }
    else
    {
        return false;
    }
}
