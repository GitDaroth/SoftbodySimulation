#include "Widget/MainWindow.h"
#include <QtWidgets/QApplication>
#include <QSurfaceFormat>

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QApplication::setStyle("windowsvista");

	QSurfaceFormat surfaceFormat;
	surfaceFormat.setDepthBufferSize(24);
	surfaceFormat.setStencilBufferSize(8);
	surfaceFormat.setSamples(16);
	surfaceFormat.setVersion(3, 3);
	surfaceFormat.setProfile(QSurfaceFormat::CoreProfile);
	surfaceFormat.setRenderableType(QSurfaceFormat::OpenGL);
	QSurfaceFormat::setDefaultFormat(surfaceFormat);

	MainWindow w;
	w.show();
	return app.exec();
}

