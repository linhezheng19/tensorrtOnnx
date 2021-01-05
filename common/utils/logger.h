/**
 * Created by linhezheng
 * Basic tensor rt engine API logger.
 * 2020/09/01
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>

using namespace std;

namespace logger {
enum class LEVEL {
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    Logger() {};
    ~Logger() {};

    template<typename T, typename T1, typename T2>
    void logger(const T& prefix_msg, const T1& msg = "",  const T2& msg2 = "", logger::LEVEL level=logger::LEVEL::INFO) {
        switch(level) {
            case(logger::LEVEL::INFO):
                cout << "[--INFO---] : " << prefix_msg << "  " << msg << "  " << msg2 << endl;
                break;
            case(logger::LEVEL::WARNING):
                cout << "[--WARN---] : " << prefix_msg << "  " << msg << "  " << msg2 << endl;
                break;
            case(logger::LEVEL::ERROR):
                cout << "[--ERROR--] : " << prefix_msg << "  " << msg << "  " << msg2 << endl;
                break;
            default:
                break;
        }
        return;
    };

    template<typename T, typename T1>
    void logger(const T& prefix_msg, const T1& msg = "", logger::LEVEL level=logger::LEVEL::INFO) {
        switch(level) {
            case(logger::LEVEL::INFO):
                cout << "[--INFO---] : " << prefix_msg << "  " << msg << endl;
                break;
            case(logger::LEVEL::WARNING):
                cout << "[--WARN---] : " << prefix_msg << "  " << msg << endl;
                break;
            case(logger::LEVEL::ERROR):
                cout << "[--ERROR--] : " << prefix_msg << "  " << msg << endl;
                break;
            default:
                break;
        }
        return;
    };

    template<typename T>
    void logger(const T& msg, logger::LEVEL level=logger::LEVEL::INFO) {
        switch(level) {
            case(logger::LEVEL::INFO):
                cout << "[--INFO---] : " << msg << endl;
                break;
            case(logger::LEVEL::WARNING):
                cout << "[--WARN---] : " << msg << endl;
                break;
            case(logger::LEVEL::ERROR):
                cout << "[--ERROR--] : " << msg << endl;
                break;
            default:
                break;
        }
        return;
    };

};
}

#endif  // LOGGER_H
