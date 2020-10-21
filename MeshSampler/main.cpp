#include <QtCore/QCoreApplication>
#include <QVector3D>
#include <QVector>
#include <QRandomGenerator>
#include <iostream>
#include <QDebug>
#include <QString>
#include <QStringList>
#include <QFile>
#include <QDataStream>
#include <QTextStream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

bool doesRayHitTriangle(
	const QVector3D& orig, const QVector3D& dir,
	const QVector3D& v0, const QVector3D& v1, const QVector3D& v2)
{
	// compute plane's normal
	QVector3D v0v1 = v1 - v0;
	QVector3D v0v2 = v2 - v0;
	QVector3D N = QVector3D::crossProduct(v0v1, v0v2); // N 
	N.normalize();

	// check if ray and plane are parallel ?
	float NdotRayDirection = QVector3D::dotProduct(N, dir);
	if (fabs(NdotRayDirection) < 0.000001f) // almost 0 
		return false; // they are parallel so they don't intersect ! 

	// compute t (equation 3)
	float t = (QVector3D::dotProduct(N, v0 - orig)) / NdotRayDirection;
	// check if the triangle is in behind the ray
	if (t < 0) 
		return false; // the triangle is behind 

	QVector3D P = orig + t * dir;

	QVector3D w = P - v0;
	QVector3D u = v0v1;
	QVector3D v = v0v2;

	float uu = QVector3D::dotProduct(u, u);
	float uv = QVector3D::dotProduct(u, v);
	float vv = QVector3D::dotProduct(v, v);
	float wu = QVector3D::dotProduct(w, u);
	float wv = QVector3D::dotProduct(w, v);
	
	float denominator = (uv * uv - uu * vv);
	if (denominator == 0.f)
		return false;

	float q = (uv * wv - vv * wu) / denominator;
	float r = (uv * wu - uu * wv) / denominator;

	if ((q > 0.f) && (r > 0.f) && (q + r < 1.f))
		return true;
	else
		return false;
}

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	QString meshName = "octopus2";
	float voxelSize = 0.1f;
	float scale = 1.f;

	// mesh reinladen
	std::string inputfile = meshName.toStdString() + ".obj";
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

	if (!warn.empty()) {
		std::cout << warn << std::endl;
	}
	if (!err.empty()) {
		std::cerr << err << std::endl;
	}
	if (!ret) {
		exit(1);
	}

	float minX = 9999999.f;
	float maxX = -9999999.f;
	float minY = 9999999.f;
	float maxY = -9999999.f;
	float minZ = 9999999.f;
	float maxZ = -9999999.f;
	// bounding box des meshes berechnen
	for (int i = 0; i < attrib.vertices.size() / 3; i++)
	{
		float x = scale * attrib.vertices[3 * i + 0];
		if (x < minX)
			minX = x;
		if (x > maxX)
			maxX = x;

		float y = scale * attrib.vertices[3 * i + 1];
		if (y < minY)
			minY = y;
		if (y > maxY)
			maxY = y;

		float z = scale * attrib.vertices[3 * i + 2];
		if (z < minZ)
			minZ = z;
		if (z > maxZ)
			maxZ = z;
	}

	minX *= 1.1f;
	maxX *= 1.1f;
	minY *= 1.1f;
	maxY *= 1.1f;
	minZ *= 1.1f;
	maxZ *= 1.1f;

	qDebug() << "x:" << minX << maxX;
	qDebug() << "y:" << minY << maxY;
	qDebug() << "z:" << minZ << maxZ;

	QVector<QVector3D> particlePositionsInsideMesh;

	for (float x = minX; x <= maxX; x += voxelSize)
	{
		for (float y = minY; y <= maxY; y += voxelSize)
		{
			for (float z = minZ; z <= maxZ; z += voxelSize)
			{
				QVector3D samplePoint(x, y, z);
				float randX = (float)QRandomGenerator::global()->bounded(-10000, 10000) / 10000.f;
				float randY = (float)QRandomGenerator::global()->bounded(-10000, 10000) / 10000.f;
				float randZ = (float)QRandomGenerator::global()->bounded(-10000, 10000) / 10000.f;
				QVector3D rayDirection(randX, randY, randZ);
				rayDirection.normalize();

				// schleife durch alle dreiecke und punkt mit ray testen 
				int triangleHitCount = 0;
				size_t index_offset = 0;
				for (size_t f = 0; f < shapes[0].mesh.num_face_vertices.size(); f++)
				{
					tinyobj::index_t idx = shapes[0].mesh.indices[index_offset + 0];
					tinyobj::real_t v0x = scale * attrib.vertices[3 * idx.vertex_index + 0];
					tinyobj::real_t v0y = scale * attrib.vertices[3 * idx.vertex_index + 1];
					tinyobj::real_t v0z = scale * attrib.vertices[3 * idx.vertex_index + 2];
					QVector3D v0(v0x, v0y, v0z);

					idx = shapes[0].mesh.indices[index_offset + 1];
					tinyobj::real_t v1x = scale * attrib.vertices[3 * idx.vertex_index + 0];
					tinyobj::real_t v1y = scale * attrib.vertices[3 * idx.vertex_index + 1];
					tinyobj::real_t v1z = scale * attrib.vertices[3 * idx.vertex_index + 2];
					QVector3D v1(v1x, v1y, v1z);

					idx = shapes[0].mesh.indices[index_offset + 2];
					tinyobj::real_t v2x = scale * attrib.vertices[3 * idx.vertex_index + 0];
					tinyobj::real_t v2y = scale * attrib.vertices[3 * idx.vertex_index + 1];
					tinyobj::real_t v2z = scale * attrib.vertices[3 * idx.vertex_index + 2];
					QVector3D v2(v2x, v2y, v2z);

					index_offset += 3;

					if (doesRayHitTriangle(samplePoint, rayDirection, v0, v1, v2))
						triangleHitCount++;
				}
				
				if (triangleHitCount % 2 == 1) // odd number of hits
					particlePositionsInsideMesh.append(samplePoint);	
			}
		}
	}

	QFile file(meshName + ".pts");
	file.open(QIODevice::WriteOnly);
	file.resize(0);
	QDataStream out(&file);
	out << particlePositionsInsideMesh.size();
	for (QVector3D particlePosition : particlePositionsInsideMesh)
	{
		out << particlePosition.x() << particlePosition.y() << particlePosition.z();
		//qDebug() << particlePosition.x() << particlePosition.y() << particlePosition.z();
	}

	file.close();

	qDebug() << "Particles:" << particlePositionsInsideMesh.size();
	qDebug() << "Finished";





	QVector<QString> startLines;
	QVector<QString> vertexLines;
	QVector<QString> normalLines;
	QVector<QString> fixedNormalLines;
	QVector<QString> smoothLines;
	QVector<QString> faceLines;
	QVector<QString> fixedFaceLines;

	QFile objFile(meshName + ".obj");
	objFile.open(QIODevice::ReadOnly);
	QTextStream in(&objFile);
	startLines.push_back(in.readLine());
	startLines.push_back(in.readLine());
	startLines.push_back(in.readLine());
	while (!in.atEnd())
	{
		QString line = in.readLine();
		if (line[0] == 'v' && line[1] == 'n')
			normalLines.push_back(line);
		else if (line[0] == 'v')
			vertexLines.push_back(line);
		else if (line[0] == 's')
			smoothLines.push_back(line);
		else if (line[0] == 'f')
			faceLines.push_back(line);
	}
	objFile.close();

	qDebug() << "Vertices:" << vertexLines.size();
	qDebug() << "Normals:" << normalLines.size();

	fixedNormalLines.resize(normalLines.size());
	for (QString faceLine : faceLines)
	{
		QString fixedFaceLine = "f";
		QStringList faceList = faceLine.split(" ");
		for (int i = 1; i < 4; i++)
		{
			QStringList vertexIndices = faceList[i].split("//");
			int positionIndex = vertexIndices[0].toInt();
			int normalIndex = vertexIndices[1].toInt();

			fixedNormalLines[positionIndex - 1] = normalLines[normalIndex - 1];
			fixedFaceLine += " " + QString::number(positionIndex) + "//" + QString::number(positionIndex);
		}
		fixedFaceLines.append(fixedFaceLine);
	}


	QFile outFile(meshName + ".obj");
	outFile.open(QIODevice::WriteOnly);
	outFile.resize(0);
	QTextStream outStream(&outFile);

	for (QString startLine : startLines)
		outStream << startLine << "\n";
	for (QString vertexLine : vertexLines)
		outStream << vertexLine << "\n";
	for (QString fixedNormalLine : fixedNormalLines)
		outStream << fixedNormalLine << "\n";
	for (QString smoothLine : smoothLines)
		outStream << smoothLine << "\n";
	for (QString fixedFaceLine : fixedFaceLines)
		outStream << fixedFaceLine << "\n";

	outFile.close();
	qDebug() << "Finished";


	return a.exec();
}
