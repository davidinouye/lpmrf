#ifndef _UTILS_H // Include guard
#define _UTILS_H
#include <stdarg.h>

// Simple print variable macro
#define varprint(a) std::cout << #a << ":" << std::endl << a << std::endl;

// Base code from  http://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf   
void message(const double level, const double verbosity, const std::string fmt, ...) {
    if( level > verbosity ) { return; } // Skip if higher than verbosity
    int size = ((int)fmt.size()) * 2 + 50;
    std::string str;
    va_list ap;
    while (1) {     // Maximum two passes on a POSIX system...
        str.resize(size);
        va_start(ap, fmt);
        int n = vsnprintf((char *)str.data(), size, fmt.c_str(), ap);
        va_end(ap);
        if (n > -1 && n < size) {  // Everything worked
            str.resize(n);
            break;
        }
        if (n > -1)  // Needed size returned
            size = n + 1;   // For null char
        else
            size *= 2;      // Guess at a larger size (OS specific)
    }

    std::cout << str << std::endl;
}


#endif
