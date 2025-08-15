package com.ssafy.ssashinsa.heyfy.authentication.service;

import com.ssafy.ssashinsa.heyfy.authentication.entity.Users;
import com.ssafy.ssashinsa.heyfy.authentication.repository.UserRepository;
import com.ssafy.ssashinsa.heyfy.common.CustomException;
import com.ssafy.ssashinsa.heyfy.common.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.Collections;

@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {
    private final UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        System.out.println("탐색 로직");
        // 현재 db에서 데이터 가져오도록 설정
        Users user = userRepository.findByUsername(username).orElseThrow(() -> new CustomException(ErrorCode.LOGIN_FAILED));

        System.out.println("유저 불러옴 "+user.getUserId());
        return new User(
                user.getUsername(),
                user.getPassword(),
                Collections.emptyList()
        );
    }
}