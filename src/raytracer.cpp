#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <vector>


// set initial values
double near = 1, left = 1, right = -1, bottom = -1, top = 1;

// Output in P6 format, a binary file containing:
// P6
// ncolumns nrows
// Max colour value
// colours in binary format thus unreadable
void save_imageP6(int Width, int Height, const char* fname, unsigned char* pixels) {
    FILE* fp;
    const int maxVal = 255;

    fp = fopen(fname, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", Width, Height);
    fprintf(fp, "%d\n", maxVal);

    for (int j = 0; j < Height; j++) {
        fwrite(&pixels[j * Width * 3], 3, Width, fp);
    }

    fclose(fp);
}


const int bounceCount = 3;

struct RGB {
    double r, g, b;
};


struct vec3 {
    double x, y, z;

    double norm() {
        return sqrtf(powf(x, 2.0f) + powf(y, 2.0) + powf(z, 2.0f));
    }

    vec3 operator*(double scalar) {
        return vec3{ x * scalar, y * scalar, z * scalar };
    }

    vec3 operator+(vec3 vector) {
        return vec3{ x + vector.x, y + vector.y, z + vector.z };
    }

    vec3 operator-(vec3 vector) {
        return vec3{ x - vector.x, y - vector.y, z - vector.z };
    }

    vec3 unit() {
        double len = norm();
        return vec3{ x / len, y / len, z / len };
    }

    // Perform component-wise multiplication
    vec3 componentMultiply(vec3 vector) {
        return vec3{ x * vector.x, y * vector.y, z * vector.z };
    }

    vec3 clampAll(double min, double max) {
        if (x > max) x = max; else if (x < min) x = min;
        if (y > max) y = max; else if (y < min) y = min;
        if (z > max) z = max; else if (z < min) z = min;
        return vec3{ x,y,z };
    }

    /**
     * @brief Dot product of this vector and parameter vector
     *
     * @param vector
     * @return double
     */
    double dot(vec3 vector) {
        return x * vector.x + y * vector.y + z * vector.z;
    }

};

struct vec4 {
    double x, y, z, w;
    vec4 asPoint(vec3 pt) {
        x = pt.x;
        y = pt.y;
        z = pt.z;
        w = 1.f;
        return *this;
    }
    vec4 asVector(vec3 vec) {
        x = vec.x;
        y = vec.y;
        z = vec.z;
        w = 0.f;
        return *this;
    }
    double dot(vec4 vector) {
        return x * vector.x + y * vector.y + z * vector.z + w * vector.w;
    }

    vec3 asVec3() {
        return vec3{ x, y, z };
    }

    vec4 multiplyMatrix(double m[4][4]) {
        vec4 mRow1{ m[0][0], m[0][1], m[0][2], m[0][3] };
        vec4 mRow2{ m[1][0], m[1][1], m[1][2], m[1][3] };
        vec4 mRow3{ m[2][0], m[2][1], m[2][2], m[2][3] };
        vec4 mRow4{ m[3][0], m[3][1], m[3][2], m[3][3] };

        return vec4{ dot(mRow1), dot(mRow2), dot(mRow3), dot(mRow4) };
    }

    vec4 multiplyTranspose(double m[4][4]) {
        vec4 mRow1{ m[0][0], m[1][0], m[2][0], m[3][0] };
        vec4 mRow2{ m[0][1], m[1][1], m[2][1], m[3][1] };
        vec4 mRow3{ m[0][2], m[1][2], m[2][2], m[3][2] };
        vec4 mRow4{ m[0][3], m[1][3], m[2][3], m[3][3] };

        return vec4{ dot(mRow1), dot(mRow2), dot(mRow3), dot(mRow4) };
    }
};



struct vec2 {
    double x, y;
};

vec2 res{ 512, 512 };
vec3 ambientIntensity = { 0.2f, 0.2f, 0.2f };
vec3 backgroundColour = { 0.3f, 0.3f, 0.3f };

// In local space all spheres are unit size
struct Sphere {
    std::string name;
    vec3 scale{ 1.0, 1.0, 1.0 };
    vec3 position{ 0.0f, 0.0f, 0.0f };
    double radius = 1.f;
    double matrix[4][4]{ 0 };
    double inverseMatrix[4][4]{ 0 };

    // Respectively, the ambient, diffuse, specular and reflective coefficients
    double k_a, k_d, k_s, k_r;
    vec3 objectColour;
    double specularExponent;

    Sphere() {};
    Sphere(vec3 scale, vec3 posIn) {
        inverseMatrix[0][0] = 1.f / scale.x;
        inverseMatrix[1][1] = 1.f / scale.y;
        inverseMatrix[2][2] = 1.f / scale.z;
        inverseMatrix[0][3] = -posIn.x;
        inverseMatrix[1][3] = -posIn.y;
        inverseMatrix[2][3] = -posIn.z;
        inverseMatrix[3][3] = 1.0;
        position = posIn;
    }

    void computeInverse() {
        matrix[0][0] = scale.x;
        matrix[1][1] = scale.y;
        matrix[2][2] = scale.z;
        matrix[0][3] = position.x;
        matrix[1][3] = position.y;
        matrix[2][3] = position.z;
        matrix[3][3] = 1.0;

        inverseMatrix[0][0] = 1.f / scale.x;
        inverseMatrix[1][1] = 1.f / scale.y;
        inverseMatrix[2][2] = 1.f / scale.z;
        inverseMatrix[0][3] = -(position.x / scale.x);
        inverseMatrix[1][3] = -position.y / scale.y;
        inverseMatrix[2][3] = -position.z / scale.z;
        inverseMatrix[3][3] = 1.0;
    }

    /**
     * @brief Given a (assumed) world space intersection
     * with this sphere, returns the normal vector
     *
     * @param intersectWS
     * @return vec3
     */
    vec3 getNormalFromIntersect(vec3 intersectLocalSpace) {
        return intersectLocalSpace.unit();
    }

};

struct LightSource {
    vec3 location;
    vec3 intensity;
};


struct Ray {
private:
    // Ensure direction is private so it can only be set by the setter (ensuring unit length)
    vec3 direction;
public:
    vec3 startingPoint;

    Ray(vec3 rayDirection, vec3 rayStartingPoint) {
        direction = rayDirection.unit();
        startingPoint = rayStartingPoint;
    }

    void setDirection(vec3 dir) {
        direction = dir.unit();
    }

    vec3 getDirection() {
        return direction;
    }

    struct IntersectResult {
        bool intersect = false;
        // Intersection in world space
        vec3 closestIntersection;
        // The t value r(t) for the nearest intersection. Used for
        // depth comparisons 
        double closestTValue;

        // Intersection normal in world space
        // The intersection may be on the inside of the sphere in which case
        // the ray is bouncing around inside of the sphere
        vec3 intersectionNormal;

        Sphere intersectedSphere;

    };

    IntersectResult testIntersection(Sphere sphere, bool ignoreCloseIntersections = false, bool initialRay = false) {
        /*

            Solve ray/sphere intersection quadratic:
            a = |dir|^2
            b = 2dir(startingPoint - sphereCentre)
            c = |startingPoint - sphereCentre|^2 - radius^2
        */

        vec4 transformed{};
        vec3 directionTransformed = transformed.asVector(direction).multiplyMatrix(sphere.inverseMatrix).asVec3().unit();
        vec3 startingPointTransformed = transformed.asPoint(startingPoint).multiplyMatrix(sphere.inverseMatrix).asVec3();
        vec3 spherePositionTransformed = { 0.f, 0.f, 0.f };

        // Unit direction vector
        double radius = 1.0f;
        double a = 1.f;
        double b = ((startingPointTransformed - spherePositionTransformed) * 2.f).dot(directionTransformed);
        double c = powf((startingPointTransformed - spherePositionTransformed).norm(), 2.f) - powf(radius, 2.f);

        double det = powf(b, 2.0) - 4 * a * c;

        IntersectResult result{};
        if (det >= 0.0f) {
            double root1 = (-b + sqrtf(det)) / (2.0 * a);
            double root2 = (-b - sqrtf(det)) / (2.0 * a);

            // Prematurely set to true, only set to false if both roots are negative (ray intersects backwards only)
            result.intersect = true;
            result.intersectedSphere = sphere;

            double threshold = ignoreCloseIntersections ? 0.001f : 0.0f;
            vec3 furthestIntersectionLS;
            vec3 furthestIntersectionWS;

            // two t's greater than or equal to zero, choose closest 
            if (root2 > threshold && root1 > threshold) {
                if (root1 < root2) {
                    result.closestIntersection = startingPointTransformed + directionTransformed * root1;
                    furthestIntersectionLS = startingPointTransformed + directionTransformed * root2;
                    result.closestTValue = root1;
                }
                else {
                    result.closestIntersection = startingPointTransformed + directionTransformed * root2;
                    furthestIntersectionLS = startingPointTransformed + directionTransformed * root1;
                    result.closestTValue = root2;
                }

            }
            else if (root2 > threshold) {
                result.closestIntersection = startingPointTransformed + directionTransformed * root2;
                furthestIntersectionLS = startingPointTransformed + directionTransformed * root2;
                result.closestTValue = root2;
            }
            else if (root1 > threshold) {
                result.closestIntersection = startingPointTransformed + directionTransformed * root1;
                furthestIntersectionLS = startingPointTransformed + directionTransformed * root1;
                result.closestTValue = root1;
            }
            else {
                result.intersect = false;
            }

            // If an intersection occurred, determine the normal of the sphere
            if (result.intersect) {
                result.intersectionNormal = transformed.asVector(sphere.getNormalFromIntersect(result.closestIntersection)).multiplyTranspose(sphere.inverseMatrix).asVec3().unit();
                result.closestIntersection = transformed.asPoint(result.closestIntersection).multiplyMatrix(sphere.matrix).asVec3();
                furthestIntersectionWS = transformed.asPoint(furthestIntersectionLS).multiplyMatrix(sphere.matrix).asVec3();
            }

            if (initialRay) {
                if (result.closestIntersection.z > -near && furthestIntersectionWS.z > -near) {
                    result.intersect = false;
                }
                else if (result.closestIntersection.z >= -near) {
                    result.closestIntersection = furthestIntersectionWS;
                    result.intersectionNormal = transformed.asVector(sphere.getNormalFromIntersect(furthestIntersectionLS)).multiplyTranspose(sphere.inverseMatrix).asVec3().unit() * -1.f;
                }
            }

        }

        return result;

    }
};

/**
 * @brief Sets the pixel colour
 *
 * @param image
 * @param x Top left is 0
 * @param y Top left is 0
 * @param colour Values should be normalized
 * @param width
 * @param height
 */
void setPixel(unsigned char* image, int x, int y, vec3 colour, int width, int height) {
    int entriesPerRow = width * 3;
    int pixelIndex = y * entriesPerRow + x * 3;
    image[pixelIndex] = colour.x * 255.0;
    image[pixelIndex + 1] = colour.y * 255.0;
    image[pixelIndex + 2] = colour.z * 255.0;
}

// For this project, the camera is fixed at the origin
// and looking down the -z axis
const vec3 cameraOrigin{ 0.f, 0.f, 0.f };

/**
 * @brief Creates a ray corresponding to a single pixel in world space
 * starting from the observer and passing through the centre of the
 * desired pixel.
 *
 * @note pixel [0,0] is the top left [res.x, res.y] is the bottom right.
 *
 * @param pixel
 * @return Ray
 */
Ray pixelToWorldRay(vec2 pixel) {
    // Determine ray in view space before undoing view transform
    // The ray starts at the observer who is at the origin of the world
    vec3 origin = { 0.0f, 0.0f, 0.0f };

    // Determine the centre of the pixel in view space:

    // Confusingly, y-axis is the horizontal (pos y left), and x-axis (pos x up) is the vertical
    double pixDX = (right - left) / res.x;
    double pixDY = (top - bottom) / res.y;

    // We subtract half the length/height of the pixel to get to the centre
    double halfPixDY = pixDY / 2.f, halfPixDX = pixDX / 2.f;

    vec3 pixelCentre{ left + (pixDX * pixel.x), top - (pixDY * pixel.y), -near };

    return Ray(pixelCentre - cameraOrigin, cameraOrigin);
}

std::vector<Sphere> spheres = {};
std::vector<LightSource> lightSources = {};


/**
 * @brief Fires a ray that tests for intersection against all spheres
 * in the scene.
 *
 * @param ray
 * @return Ray::IntersectResult
 */
Ray::IntersectResult hitTestAllSpheres(Ray ray, bool initRay) {
    Ray::IntersectResult closestResult{};
    // Determine ray intersection result
    // Setting this to 0 is okay as long as we factor in if an intersection has been found^
    double closestIntersectDistance = 0.0;
    int closestIntersectIndex = -1;
    for (int i = 0; i < spheres.size(); i++) {

        Ray::IntersectResult result = ray.testIntersection(spheres[i], true, initRay);

        if (result.intersect) {
            double intersectDistance = (result.closestIntersection - ray.startingPoint).norm();
            // If there hasn't been an intersection yet (closestIntersectIndex == -1) or we found
            // a closer intersection (result.closestTValue < closestT)
            if (closestIntersectIndex == -1 || intersectDistance < closestIntersectDistance) {
                closestIntersectIndex = i;
                closestIntersectDistance = intersectDistance;
                closestResult = result;
            }
        }
    }
    return closestResult;
}

/**
 * @brief Compute the phong illumination model for single light source
 *
 * @param V
 * @param N
 * @param L
 * @return vec3
 */
vec3 computePhongDiffuseSpecular(vec3 V, vec3 N, vec3 L, vec3 R, Sphere sphere, LightSource lightSource) {
    double dotProd = N.dot(L);
    double specularDotProd = R.dot(V);
    vec3 diffuse = (dotProd >= 0) ? lightSource.intensity.componentMultiply(sphere.objectColour) * sphere.k_d * dotProd : vec3{ 0.0f, 0.0f, 0.0f };
    vec3 specular = specularDotProd >= 0 ? lightSource.intensity * sphere.k_s * powf(R.dot(V), sphere.specularExponent) : vec3{ 0.0f, 0.0f, 0.0f };

    return diffuse + specular;
}

vec3 computePhongAmbient(Sphere sphere) {
    return ambientIntensity.componentMultiply(sphere.objectColour) * sphere.k_a;
}

double clamp(double x, double min, double max) {
    if (x > max) return max;
    else if (x < min) return min;
    return x;
}


// Compute the colour contribution by firing shadow rays to all light sources,
// if there is an unoccluded hit, then use computePhongModel to find their contribution
// inSphere refers to if the hit was on the inside of the sphere, in which case only consider lights
// in the sphere
vec3 computeLighting(vec3 surfacePoint, vec3 surfaceNormal, Sphere sphere, bool inSphere) {
    vec3 pointLightContributions{ 0.0f, 0.0f, 0.0f };
    vec3 ambient = computePhongAmbient(sphere);

    if (inSphere == false) {
        int i = 1;
    }

    for (int i = 0; i < lightSources.size(); i++) {
        Ray inSphereCheckRay{ lightSources[i].location - sphere.position, sphere.position };

        // We determine if the light and intersection point are both in the sphere or both outside of it
        Ray::IntersectResult lightCheckResult = inSphereCheckRay.testIntersection(sphere);
        if ((inSphere && !lightCheckResult.intersect) || (!inSphere && lightCheckResult.intersect)) {
            // create a shadow ray from the surface point to the light source
            vec4 transformed{};
            vec3 surfacePointTransformed = surfacePoint;
            vec3 surfaceNormalTransformed = surfaceNormal;
            vec3 lightDirection = lightSources[i].location - surfacePointTransformed;

            // compute t for when r(t) hits the light source
            double tLight = lightDirection.norm();
            Ray shadowRay{ lightDirection, surfacePointTransformed };
            Ray::IntersectResult shadowIntersectResult = hitTestAllSpheres(shadowRay, false);

            // If an intersection did not occur (!shadowIntersectResult.intersect), or if the 
            // nearest sphere intersection is further than the nearest light intersection then the light is
            // not occluded

            if (!shadowIntersectResult.intersect || shadowIntersectResult.closestTValue > tLight) {
                vec3 V = surfacePointTransformed.unit() * -1.f;
                vec3 N = surfaceNormalTransformed.unit();
                vec3 L = lightDirection.unit();
                vec3 R = (N * clamp(N.dot(L), 0.f, 1.f)) * 2.f - L;
                vec3 phongNoReflection = computePhongDiffuseSpecular(V, N, L, R, sphere, lightSources[i]);
                pointLightContributions = pointLightContributions + phongNoReflection;

            }
        }
    }
    return (pointLightContributions + ambient).clampAll(0.f, 1.f);
}

/**
 * @brief Recursively trace light rays.
 *
 * @param ray The ray being traced
 * @param prevBounceSphereIndex The index of the sphere which this ray has reflected from. -1 if initial ray
 * @param remainingBounces The number of bounces remaining
 * @return The reflected colour from the traced ray
 */
vec3 traceRay(Ray ray, int remainingBounces, bool initRay = true) {
    // Each bounce triggers a recursive call

    Ray::IntersectResult closestResult = hitTestAllSpheres(ray, initRay);

    // We now have a result for closestResult, which is either the nearest hit, or no hit

    // If an intersection occurred, continue recursion, otherwise return no colour contribution

    if (closestResult.intersect) {
        bool inSphere = closestResult.intersectionNormal.dot(closestResult.closestIntersection - closestResult.intersectedSphere.position) < 0.f;
        vec3 localColour = computeLighting(closestResult.closestIntersection, closestResult.intersectionNormal, closestResult.intersectedSphere, inSphere);
        vec3 reflectedColour{ 0.f, 0.f, 0.f };
        if (remainingBounces > 0) {
            Ray newRay{ closestResult.intersectionNormal, closestResult.closestIntersection };
            reflectedColour = traceRay(newRay, remainingBounces - 1, false) * closestResult.intersectedSphere.k_r;
        }
        return localColour + reflectedColour;
        return localColour.clampAll(0.f, 1.f);

    }
    else {
        return remainingBounces == bounceCount ? backgroundColour : vec3{ 0.0f, 0.0f, 0.0f };
    }
}


int main(int argc, char** argv) {

    std::ifstream file;
    file.open(argv[1]);

    std::string line = "";

    std::string fileName = "";

    while (file >> line) {
        if (line == "NEAR") file >> near;
        else if (line == "LEFT") file >> left;
        else if (line == "RIGHT") file >> right;
        else if (line == "BOTTOM") file >> bottom;
        else if (line == "TOP") file >> top;
        else if (line == "RES") file >> res.x >> res.y;
        else if (line == "AMBIENT") file >> ambientIntensity.x >> ambientIntensity.y >> ambientIntensity.z;
        else if (line == "BACK") file >> backgroundColour.x >> backgroundColour.y >> backgroundColour.z;
        else if (line == "SPHERE") {
            Sphere newSphere;
            file >> line;
            file >> newSphere.position.x >> newSphere.position.y >> newSphere.position.z;
            file >> newSphere.scale.x >> newSphere.scale.y >> newSphere.scale.z;
            file >> newSphere.objectColour.x >> newSphere.objectColour.y >> newSphere.objectColour.z;
            file >> newSphere.k_a >> newSphere.k_d >> newSphere.k_s >> newSphere.k_r >> newSphere.specularExponent;
            newSphere.computeInverse();
            spheres.push_back(newSphere);
        }
        else if (line == "LIGHT") {
            LightSource newLight;
            file >> line;
            file >> newLight.location.x >> newLight.location.y >> newLight.location.z;
            file >> newLight.intensity.x >> newLight.intensity.y >> newLight.intensity.z;
            lightSources.push_back(newLight);
        }

        else if (line == "OUTPUT") {
            file >> fileName;
        }
    }


    int Width = res.x;	// Move these to your setup function. The actual resolution will be
    int Height = res.y;	// specified in the input file
    char fname3[20] = "sceneP3.ppm"; //This should be set based on the input file
    char fname6[20] = "sceneP6.ppm"; //This should be set based on the input file
    unsigned char* pixels;
    // This will be your image. Note that pixels[0] is the top left of the image and
    // pixels[3*Width*Height-1] is the bottom right of the image.
    pixels = new unsigned char [3 * Width * Height] {0};


    for (double y = 0; y < res.x; y++) {
        for (double x = 0; x < res.y; x++) {
            Ray worldRay = pixelToWorldRay({ x,y });
            setPixel(pixels, x, y, traceRay(worldRay, bounceCount), res.x, res.y);
        }
        int i = y;
    }

    save_imageP6(Width, Height, ("../render/"+ fileName).data(), pixels);


    return 0;
}