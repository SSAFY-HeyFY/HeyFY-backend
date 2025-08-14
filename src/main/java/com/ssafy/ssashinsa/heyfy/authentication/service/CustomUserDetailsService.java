package com.ssafy.ssashinsa.heyfy.authentication.service;

import com.ssafy.ssashinsa.heyfy.common.CustomException;
import com.ssafy.ssashinsa.heyfy.common.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Collections;

@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {

    private final PasswordEncoder passwordEncoder;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // 현재 db가 없어 하드코딩으로 user : user, password : ssafy로 설정. 후에 변경할 예정
        if (!"user".equals(username)) {
            throw new CustomException(ErrorCode.LOGIN_FAILED);
        }

        String encodedPassword = passwordEncoder.encode("ssafy");

        return new User(
                "user",
                encodedPassword,
                Collections.emptyList()
        );
    }
}